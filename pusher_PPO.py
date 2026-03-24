__credits__ = ["Kallinteris-Andreas"]

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.stats import truncnorm

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}


class Pusher_Mass_UniformDistribution(MujocoEnv, utils.EzPickle):
    """
    Pusher environment con supporto a domain randomization su:
    - masse di tutti i body del pusher + cilindro
    - attrito del cilindro e del piano

    NOTA IMPORTANTE:
    I nomi dei geom/body devono corrispondere a quelli nel tuo XML.

    Di default assumo:
    - body oggetto/cilindro: "object"
    - geom cilindro: "object"
    - geom piano/tavolo: "floor"

    Se nel tuo XML il piano ha un altro nome (es. "table"), cambia plane_geom_name.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "pusher-v2_attrito.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 0.1,
        domain: Optional[str] = None,
        distribuzione: float = 0.2,
        object_body_name: str = "object",
        object_geom_name: str = "object",
        plane_geom_name: str = "floor",
        **kwargs,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_xml_path = os.path.join(current_dir, xml_file)

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_near_weight,
            reward_dist_weight,
            reward_control_weight,
            domain,
            distribuzione,
            object_body_name,
            object_geom_name,
            plane_geom_name,
            **kwargs,
        )

        self.domain = domain
        self.distribuzione = distribuzione

        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        self.object_body_name = object_body_name
        self.object_geom_name = object_geom_name
        self.plane_geom_name = plane_geom_name

        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            full_xml_path,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # -------------------------
        # Indici MuJoCo utili
        # -------------------------
        self.object_body_id = self._get_body_id_by_name(self.object_body_name)
        self.object_geom_id = self._get_geom_id_by_name(self.object_geom_name)
        self.plane_geom_id = self._get_geom_id_by_name(self.plane_geom_name)

        # -------------------------
        # Masse originali: come Hopper
        # body 0 = world, quindi escludiamo quello
        # -------------------------
        self.original_masses = np.copy(self.model.body_mass[1:])

        # nomi dei body, utili per debug
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody)]

        # -------------------------
        # Attriti originali
        # -------------------------
        self.original_object_friction = np.array(
            self.model.geom_friction[self.object_geom_id], dtype=np.float64
        )
        self.original_plane_friction = np.array(
            self.model.geom_friction[self.plane_geom_id], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Utility: recupero body/geom id dal nome
    # ------------------------------------------------------------------
    def _get_body_id_by_name(self, body_name: str) -> int:
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name == body_name:
                return i

        available = [self.model.body(i).name for i in range(self.model.nbody)]
        raise ValueError(
            f"Body '{body_name}' non trovato nel modello MuJoCo. "
            f"Body disponibili: {available}"
        )

    def _get_geom_id_by_name(self, geom_name: str) -> int:
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if name == geom_name:
                return i

        available = [self.model.geom(i).name for i in range(self.model.ngeom)]
        raise ValueError(
            f"Geom '{geom_name}' non trovato nel modello MuJoCo. "
            f"Geom disponibili: {available}"
        )

    # ------------------------------------------------------------------
    # Step / reward
    # ------------------------------------------------------------------
    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, False, info

    def _get_rew(self, action):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        reward = reward_dist + reward_ctrl + reward_near

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
        }

        return reward, reward_info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_model(self):
        qpos = self.init_qpos.copy()

        self.goal_pos = np.asarray([0.0, 0.0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.6, high=-0.3, size=1),
                    self.np_random.uniform(low=-0.3, high=0.3, size=1),
                ]
            )

            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0

        self.set_state(qpos, qvel)

        # Domain randomization nel source
        if self.domain == "source":
            self.set_random_parameters()

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )

    # ------------------------------------------------------------------
    # API comune per randomizzazione
    # ------------------------------------------------------------------
    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """
        Versione base:
        randomizza tutte le masse con distribuzione uniforme
        come nel tuo Hopper.
        """
        masses = np.array(self.model.body_mass[1:], dtype=np.float64)

        sampled = self.np_random.uniform(
            1 - self.distribuzione,
            1 + self.distribuzione,
            size=self.original_masses.shape,
        ) * self.original_masses

        masses[:] = sampled
        return {"masses": masses}

    def get_parameters(self):
        return {
            "masses": np.array(self.model.body_mass[1:]),
            "object_friction": np.array(self.model.geom_friction[self.object_geom_id]),
            "plane_friction": np.array(self.model.geom_friction[self.plane_geom_id]),
        }

    def set_parameters(self, params):
        if "masses" in params:
            self.model.body_mass[1:] = params["masses"]

        if "object_friction" in params:
            self.model.geom_friction[self.object_geom_id] = params["object_friction"]

        if "plane_friction" in params:
            self.model.geom_friction[self.plane_geom_id] = params["plane_friction"]


# ----------------------------------------------------------------------
# SOLO MASSA - GAUSSIANA TRONCATA
# ----------------------------------------------------------------------
class Pusher_Mass_GaussianTruncated(Pusher_Mass_UniformDistribution):
    def sample_parameters(self):
        mean = self.original_masses
        std = 0.1 * mean

        a = ((1.0 - self.distribuzione) * mean - mean) / (std + 1e-6)
        b = ((1.0 + self.distribuzione) * mean - mean) / (std + 1e-6)

        sampled = truncnorm.rvs(
            a, b, loc=mean, scale=std, random_state=self.np_random
        )

        return {"masses": sampled}


# ----------------------------------------------------------------------
# SOLO ATTRITO - UNIFORME
# ----------------------------------------------------------------------
class Pusher_OnlyFriction_Uniform(Pusher_Mass_UniformDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.original_object_friction = np.copy(
            self.model.geom_friction[self.object_geom_id]
        )
        self.original_plane_friction = np.copy(
            self.model.geom_friction[self.plane_geom_id]
        )

        # Manteniamo le masse originali in questa variante
        self.model.body_mass[1:] = self.original_masses

    def sample_parameters(self):
        low, high = 1.0 - self.distribuzione, 1.0 + self.distribuzione

        new_object_friction = self.original_object_friction * self.np_random.uniform(
            low, high, size=self.original_object_friction.shape
        )
        new_plane_friction = self.original_plane_friction * self.np_random.uniform(
            low, high, size=self.original_plane_friction.shape
        )

        return {
            "object_friction": new_object_friction,
            "plane_friction": new_plane_friction,
        }

    def set_parameters(self, params):
        self.model.geom_friction[self.object_geom_id] = params["object_friction"]
        self.model.geom_friction[self.plane_geom_id] = params["plane_friction"]

        # Forziamo le masse a restare originali
        self.model.body_mass[1:] = self.original_masses

    def get_parameters(self):
        return {
            "object_friction": np.array(self.model.geom_friction[self.object_geom_id]),
            "plane_friction": np.array(self.model.geom_friction[self.plane_geom_id]),
        }


# ----------------------------------------------------------------------
# SOLO ATTRITO - GAUSSIANA TRONCATA
# ----------------------------------------------------------------------
class Pusher_Friction_Gaussian(Pusher_OnlyFriction_Uniform):
    def sample_parameters(self):
        mean_obj = self.original_object_friction
        std_obj = 0.1 * mean_obj

        a_obj = ((1.0 - self.distribuzione) * mean_obj - mean_obj) / (std_obj + 1e-6)
        b_obj = ((1.0 + self.distribuzione) * mean_obj - mean_obj) / (std_obj + 1e-6)

        new_object_friction = truncnorm.rvs(
            a_obj, b_obj, loc=mean_obj, scale=std_obj, random_state=self.np_random
        )

        mean_plane = self.original_plane_friction
        std_plane = 0.1 * mean_plane

        a_plane = ((1.0 - self.distribuzione) * mean_plane - mean_plane) / (std_plane + 1e-6)
        b_plane = ((1.0 + self.distribuzione) * mean_plane - mean_plane) / (std_plane + 1e-6)

        new_plane_friction = truncnorm.rvs(
            a_plane, b_plane, loc=mean_plane, scale=std_plane, random_state=self.np_random
        )

        return {
            "object_friction": new_object_friction,
            "plane_friction": new_plane_friction,
        }


# ----------------------------------------------------------------------
# MASSA + ATTRITO - UNIFORME
# ----------------------------------------------------------------------
class Pusher_MassAndFriction_Uniform(Pusher_Mass_UniformDistribution):
    def sample_parameters(self):
        low, high = 1.0 - self.distribuzione, 1.0 + self.distribuzione

        new_masses = self.original_masses * self.np_random.uniform(
            low, high, size=self.original_masses.shape
        )

        new_object_friction = self.original_object_friction * self.np_random.uniform(
            low, high, size=self.original_object_friction.shape
        )
        new_plane_friction = self.original_plane_friction * self.np_random.uniform(
            low, high, size=self.original_plane_friction.shape
        )

        return {
            "masses": new_masses,
            "object_friction": new_object_friction,
            "plane_friction": new_plane_friction,
        }

    def get_parameters(self):
        return {
            "masses": np.array(self.model.body_mass[1:]),
            "object_friction": np.array(self.model.geom_friction[self.object_geom_id]),
            "plane_friction": np.array(self.model.geom_friction[self.plane_geom_id]),
        }


# ----------------------------------------------------------------------
# MASSA + ATTRITO - GAUSSIANA TRONCATA
# ----------------------------------------------------------------------
class Pusher_MassAndFriction_Gaussian(Pusher_MassAndFriction_Uniform):
    def sample_parameters(self):
        # Masse
        mean_m = self.original_masses
        std_m = 0.1 * mean_m
        a_m = ((1.0 - self.distribuzione) * mean_m - mean_m) / (std_m + 1e-6)
        b_m = ((1.0 + self.distribuzione) * mean_m - mean_m) / (std_m + 1e-6)

        new_masses = truncnorm.rvs(
            a_m, b_m, loc=mean_m, scale=std_m, random_state=self.np_random
        )

        # Attrito oggetto
        mean_obj = self.original_object_friction
        std_obj = 0.1 * mean_obj
        a_obj = ((1.0 - self.distribuzione) * mean_obj - mean_obj) / (std_obj + 1e-6)
        b_obj = ((1.0 + self.distribuzione) * mean_obj - mean_obj) / (std_obj + 1e-6)

        new_object_friction = truncnorm.rvs(
            a_obj, b_obj, loc=mean_obj, scale=std_obj, random_state=self.np_random
        )

        # Attrito piano
        mean_plane = self.original_plane_friction
        std_plane = 0.1 * mean_plane
        a_plane = ((1.0 - self.distribuzione) * mean_plane - mean_plane) / (std_plane + 1e-6)
        b_plane = ((1.0 + self.distribuzione) * mean_plane - mean_plane) / (std_plane + 1e-6)

        new_plane_friction = truncnorm.rvs(
            a_plane, b_plane, loc=mean_plane, scale=std_plane, random_state=self.np_random
        )

        return {
            "masses": new_masses,
            "object_friction": new_object_friction,
            "plane_friction": new_plane_friction,
        }


# ──────────────────────────────────────────────
# AMBIENTI TARGET FISSI (evaluation)
# ──────────────────────────────────────────────

class Pusher_Target_Mass(Pusher_Mass_UniformDistribution):
    """Target con sole masse scalate di un fattore fisso."""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale
        self.model.body_mass[1:] = self.original_masses * self.scale

    def reset_model(self):
        qpos = self.init_qpos.copy()

        self.goal_pos = np.asarray([0.0, 0.0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.6, high=-0.3, size=1),
                    self.np_random.uniform(low=-0.3, high=0.3, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        self.model.body_mass[1:] = self.original_masses * self.scale
        self.model.geom_friction[self.object_geom_id] = self.original_object_friction
        self.model.geom_friction[self.plane_geom_id] = self.original_plane_friction

        return self._get_obs()


class Pusher_Target_Friction(Pusher_Mass_UniformDistribution):
    """Target con soli attriti scalati di un fattore fisso."""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale

        self.model.geom_friction[self.object_geom_id] = self.original_object_friction * self.scale
        self.model.geom_friction[self.plane_geom_id] = self.original_plane_friction * self.scale

    def reset_model(self):
        qpos = self.init_qpos.copy()

        self.goal_pos = np.asarray([0.0, 0.0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.6, high=-0.3, size=1),
                    self.np_random.uniform(low=-0.3, high=0.3, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        self.model.body_mass[1:] = self.original_masses
        self.model.geom_friction[self.object_geom_id] = self.original_object_friction * self.scale
        self.model.geom_friction[self.plane_geom_id] = self.original_plane_friction * self.scale

        return self._get_obs()


class Pusher_Target_MassAndFriction(Pusher_Mass_UniformDistribution):
    """Target con masse e attriti scalati di un fattore fisso."""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale

        self.model.body_mass[1:] = self.original_masses * self.scale
        self.model.geom_friction[self.object_geom_id] = self.original_object_friction * self.scale
        self.model.geom_friction[self.plane_geom_id] = self.original_plane_friction * self.scale

    def reset_model(self):
        qpos = self.init_qpos.copy()

        self.goal_pos = np.asarray([0.0, 0.0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.6, high=-0.3, size=1),
                    self.np_random.uniform(low=-0.3, high=0.3, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos

        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)

        self.model.body_mass[1:] = self.original_masses * self.scale
        self.model.geom_friction[self.object_geom_id] = self.original_object_friction * self.scale
        self.model.geom_friction[self.plane_geom_id] = self.original_plane_friction * self.scale

        return self._get_obs()


"""
    Registered environments for Pusher Domain Randomization
"""

variations = [0.1, 0.2, 0.5, 0.8]
max_steps = 500

# 1. AMBIENTI STANDARD
gym.register(
    id="CustomPusher-v0",
    entry_point=f"{__name__}:Pusher_Mass_UniformDistribution",
    max_episode_steps=max_steps,
    kwargs={"xml_file": "pusher-v2.xml"},
)

gym.register(
    id="CustomPusher-source-v0",
    entry_point=f"{__name__}:Pusher_Mass_UniformDistribution",
    max_episode_steps=max_steps,
    kwargs={
        "xml_file": "pusher-v2_attrito.xml",
        "domain": "source",
    },
)

gym.register(
    id="CustomPusher-target-v0",
    entry_point=f"{__name__}:Pusher_Mass_UniformDistribution",
    max_episode_steps=max_steps,
    kwargs={
        "xml_file": "pusher-v2_attrito.xml",
        "domain": "target",
    },
)

# 2. TUTTE LE VARIANTI CON RANDOMIZZAZIONE
registry = [
    ("Mass-Uni", "Pusher_Mass_UniformDistribution"),
    ("Mass-Gauss", "Pusher_Mass_GaussianTruncated"),
    ("Fric-Uni", "Pusher_OnlyFriction_Uniform"),
    ("Fric-Gauss", "Pusher_Friction_Gaussian"),
    ("MassFric-Uni", "Pusher_MassAndFriction_Uniform"),
    ("MassFric-Gauss", "Pusher_MassAndFriction_Gaussian"),
]

for suffix, cls in registry:
    for dist in variations:
        percent = int(dist * 100)
        gym.register(
            id=f"Pusher-{suffix}-{percent}-v0",
            entry_point=f"{__name__}:{cls}",
            max_episode_steps=max_steps,
            kwargs={
                "xml_file": "pusher-v2_attrito.xml",
                "domain": "source",
                "distribuzione": dist,
                # Se nel tuo XML il piano NON si chiama "floor", cambia qui:
                # "plane_geom_name": "table"
            },
        )

# 3. TARGET FISSI
target_configs = {
    "Easy": 0.95,
    "Medium": 1.25,
    "Hard": 1.55,
}

for difficulty, scale in target_configs.items():
    gym.register(
        id=f"Pusher-Target-Mass-{difficulty}-v0",
        entry_point=f"{__name__}:Pusher_Target_Mass",
        max_episode_steps=max_steps,
        kwargs={
            "xml_file": "pusher-v2_attrito.xml",
            "scale": scale,
        },
    )

    gym.register(
        id=f"Pusher-Target-Fric-{difficulty}-v0",
        entry_point=f"{__name__}:Pusher_Target_Friction",
        max_episode_steps=max_steps,
        kwargs={
            "xml_file": "pusher-v2_attrito.xml",
            "scale": scale,
        },
    )

    gym.register(
        id=f"Pusher-Target-Both-{difficulty}-v0",
        entry_point=f"{__name__}:Pusher_Target_MassAndFriction",
        max_episode_steps=max_steps,
        kwargs={
            "xml_file": "pusher-v2_attrito.xml",
            "scale": scale,
        },
    )