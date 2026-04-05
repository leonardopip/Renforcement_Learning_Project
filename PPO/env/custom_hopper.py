"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.stats import truncnorm

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}

# Variano Uniforme delle masse dei joint
class Hopper_Mass_UniformDistribution(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        distribuzione: float = 0.2,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            domain,
            distribuzione,
            **kwargs,
        ) 
        
        self.domain = domain
        self.distribuzione = distribuzione
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if xml_file == "hopper.xml":
             xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
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

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        self.original_masses = np.copy(self.model.body_mass[1:])    # Default link masses

        self.body_names = [self.model.body(i).name for i in range(1, len(self.model.body_mass))]

        
        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.model.body_mass[1] -= 1.0
            
            


    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
 
        # Custom domain randomization
        if self.domain == 'source':
            self.set_random_parameters() # TODO: May be useful insert a parameter to control if to sample new mass params

            #print("Nuove masse:", self.model.body_mass[1:]) 



        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        N.B. You can't change the mass of the torso (first link)
        TODO
        """

        masses = np.array(self.model.body_mass[1:], dtype=np.float64)
        sampled_others = self.np_random.uniform(1-self.distribuzione, 1+self.distribuzione )* self.original_masses[1:]
        masses[1:] = sampled_others
        
        return masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.model.body_mass[1:] = task


# distribuzione gaussiana tronccta con cui variano le masse dei joint del hopper
class Hopper_Mass_GaussianTroncata(Hopper_Mass_UniformDistribution):
    """Eredita tutto dalla Uniforme, cambia solo il campionamento"""
    def sample_parameters(self):
        mean = self.original_masses
        std = 0.1 * mean
        # Usiamo self.distribuzione per definire i bound della troncata
        a = ((1.0 - self.distribuzione) * mean - mean) / std
        b = ((1.0 + self.distribuzione) * mean - mean) / std
        
        sampled = truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.np_random)
        return sampled
class Hopper_OnlyFriction_uniform(Hopper_Mass_UniformDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assicuriamoci che le masse siano quelle originali
        self.model.body_mass[1:] = self.original_masses
        self.original_frictions = np.copy(self.model.geom_friction)

    def sample_parameters(self):
        # Genera solo i nuovi attriti
        low, high = 1.0 - self.distribuzione, 1.0 + self.distribuzione
        return self.original_frictions * self.np_random.uniform(low, high, size=self.original_frictions.shape)

    def set_parameters(self, frictions):
        # Modifica SOLO gli attriti
        self.model.geom_friction[:] = frictions
        # Opzionale: forziamo le masse a restare originali ad ogni reset
        self.model.body_mass[1:] = self.original_masses

    def get_parameters(self):
        return np.array(self.model.geom_friction)
    
class Hopper_MassAndFriction_Uniform(Hopper_Mass_UniformDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_frictions = np.copy(self.model.geom_friction)

    def sample_parameters(self):
        low, high = 1.0 - self.distribuzione, 1.0 + self.distribuzione
        
        # Campionamento Masse
        new_masses = self.original_masses * self.np_random.uniform(low, high, size=self.original_masses.shape)
        
        # Campionamento Attriti
        new_frictions = self.original_frictions * self.np_random.uniform(low, high, size=self.original_frictions.shape)
        
        return {"masses": new_masses, "frictions": new_frictions}

    def set_parameters(self, params):
        # Spacchettiamo il dizionario e applichiamo entrambi
        self.model.body_mass[1:] = params["masses"]
        self.model.geom_friction[:] = params["frictions"]

    def get_parameters(self):
        return {
            "masses": np.array(self.model.body_mass[1:]),
            "frictions": np.array(self.model.geom_friction)
        }
    
class Hopper_Friction_Gaussian(Hopper_OnlyFriction_uniform):
    def sample_parameters(self):
        mean = self.original_frictions
        std = 0.1 * mean  # Deviazione standard al 10% della media
        
        # Calcolo dei bound per la troncata basati su self.distribuzione
        a = ((1.0 - self.distribuzione) * mean - mean) / (std + 1e-6)
        b = ((1.0 + self.distribuzione) * mean - mean) / (std + 1e-6)
        
        sampled_frictions = truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.np_random)
        return sampled_frictions
    
class Hopper_MassAndFriction_Gaussian(Hopper_MassAndFriction_Uniform):
    def sample_parameters(self):
        # 1. Campionamento Masse
        mean_m = self.original_masses
        std_m = 0.1 * mean_m
        a_m = ((1.0 - self.distribuzione) * mean_m - mean_m) / (std_m + 1e-6)
        b_m = ((1.0 + self.distribuzione) * mean_m - mean_m) / (std_m + 1e-6)
        new_masses = truncnorm.rvs(a_m, b_m, loc=mean_m, scale=std_m, random_state=self.np_random)

        # 2. Campionamento Attriti
        mean_f = self.original_frictions
        std_f = 0.1 * mean_f
        a_f = ((1.0 - self.distribuzione) * mean_f - mean_f) / (std_f + 1e-6)
        b_f = ((1.0 + self.distribuzione) * mean_f - mean_f) / (std_f + 1e-6)
        new_frictions = truncnorm.rvs(a_f, b_f, loc=mean_f, scale=std_f, random_state=self.np_random)

        return {"masses": new_masses, "frictions": new_frictions}


# ──────────────────────────────────────────────
# AMBIENTI TARGET FISSI (per evaluation)
# ──────────────────────────────────────────────

class Hopper_Target_Mass(Hopper_Mass_UniformDistribution):
    """Target con sole masse scalate di un fattore fisso"""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale
        self.model.body_mass[1:] = self.original_masses * self.scale

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        # Masse fisse ad ogni reset
        self.model.body_mass[1:] = self.original_masses * self.scale
        return self._get_obs()


class Hopper_Target_Friction(Hopper_Mass_UniformDistribution):
    """Target con soli attriti scalati di un fattore fisso"""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale
        self.original_frictions = np.copy(self.model.geom_friction)
        self.model.geom_friction[:] = self.original_frictions * self.scale

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        # Attriti fissi ad ogni reset
        self.model.geom_friction[:] = self.original_frictions * self.scale
        return self._get_obs()


class Hopper_Target_MassAndFriction(Hopper_Mass_UniformDistribution):
    """Target con masse e attriti scalati di un fattore fisso"""
    def __init__(self, scale: float = 1.0, **kwargs):
        kwargs["domain"] = "target"
        super().__init__(**kwargs)
        self.scale = scale
        self.original_frictions = np.copy(self.model.geom_friction)
        self.model.body_mass[1:] = self.original_masses * self.scale
        self.model.geom_friction[:] = self.original_frictions * self.scale

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)
        # Masse e attriti fissi ad ogni reset
        self.model.body_mass[1:] = self.original_masses * self.scale
        self.model.geom_friction[:] = self.original_frictions * self.scale
        return self._get_obs()





"""
    Registered environments for Hopper Domain Randomization
"""

variations = [0.1, 0.2, 0.5, 0.8]
max_steps = 500

# 1. AMBIENTI STANDARD
gym.register(
    id="CustomHopper-v0",
    entry_point=f"{__name__}:Hopper_Mass_UniformDistribution",
    max_episode_steps=max_steps,
)
gym.register(
    id="CustomHopper-source-v0",
    entry_point=f"{__name__}:Hopper_Mass_UniformDistribution",
    max_episode_steps=max_steps,
    kwargs={"domain": "source"}
)
gym.register(
    id="CustomHopper-target-v0",
    entry_point=f"{__name__}:Hopper_Mass_UniformDistribution",
    max_episode_steps=max_steps,
    kwargs={"domain": "target"}
)

# 2. TUTTI I VARIANTI CON RANDOMIZZAZIONE
registry = [
    # (suffisso_id,        classe)
    ("Mass-Uni",           "Hopper_Mass_UniformDistribution"),
    ("Mass-Gauss",         "Hopper_Mass_GaussianTroncata"),
    ("Fric-Uni",           "Hopper_OnlyFriction_uniform"),
    ("Fric-Gauss",         "Hopper_Friction_Gaussian"),
    ("MassFric-Uni",       "Hopper_MassAndFriction_Uniform"),
    ("MassFric-Gauss",     "Hopper_MassAndFriction_Gaussian"),
]

for suffix, cls in registry:
    for dist in variations:
        percent = int(dist * 100)
        gym.register(
            id=f"Hopper-{suffix}-{percent}-v0",
            entry_point=f"{__name__}:{cls}",
            max_episode_steps=max_steps,
            kwargs={"domain": "source", "distribuzione": dist}
        )

# Registrazione dei 9 ambienti target
target_configs = {
    "Easy":   0.95,
    "Medium": 1.25,
    "Hard":   1.55,
}

for difficulty, scale in target_configs.items():
    gym.register(
        id=f"Hopper-Target-Mass-{difficulty}-v0",
        entry_point=f"{__name__}:Hopper_Target_Mass",
        max_episode_steps=max_steps,
        kwargs={"scale": scale}
    )
    gym.register(
        id=f"Hopper-Target-Fric-{difficulty}-v0",
        entry_point=f"{__name__}:Hopper_Target_Friction",
        max_episode_steps=max_steps,
        kwargs={"scale": scale}
    )
    gym.register(
        id=f"Hopper-Target-Both-{difficulty}-v0",
        entry_point=f"{__name__}:Hopper_Target_MassAndFriction",
        max_episode_steps=max_steps,
        kwargs={"scale": scale}
    )


#Gli ambienti registrati sono quindi:

#Hopper-Target-Mass-Easy-v0    (masse × 0.95)
#Hopper-Target-Mass-Medium-v0  (masse × 1.25)
#Hopper-Target-Mass-Hard-v0    (masse × 1.55)
#Hopper-Target-Fric-Easy-v0    (attriti × 0.95)
#Hopper-Target-Fric-Medium-v0  (attriti × 1.25)
#Hopper-Target-Fric-Hard-v0    (attriti × 1.55)
#Hopper-Target-Both-Easy-v0    (masse+attriti × 0.95)
#Hopper-Target-Both-Medium-v0  (masse+attriti × 1.25)
#Hopper-Target-Both-Hard-v0    (masse+attriti × 1.55)