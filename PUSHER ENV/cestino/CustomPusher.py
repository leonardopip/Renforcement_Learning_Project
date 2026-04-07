import os
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Dict, Union, Optional, Tuple

DEFAULT_CAMERA_CONFIG = {"trackbodyid": -1, "distance": 4.0}

class CustomPusher(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file: str = "pusher_v5.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1.0,
        reward_control_weight: float = 0.1,
        reset_noise_scale: float = 5e-3,
        domain: Optional[str] = None,
        **kwargs,
    ):
        # --- stesso pattern del CustomHopper ---
        self.custom_domain_randomization = kwargs.pop('dr_active', False)
        self.dr_percentage = kwargs.pop('dr_percentage', 0.2)

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_near_weight,
            reward_dist_weight,
            reward_control_weight,
            reset_noise_scale,
            domain,
            **kwargs,
        )

        self._reward_near_weight    = reward_near_weight
        self._reward_dist_weight    = reward_dist_weight
        self._reward_control_weight = reward_control_weight
        self._reset_noise_scale     = reset_noise_scale

        # risolvi il path dell'XML (stesso trucco del CustomHopper)
        if xml_file == "pusher_v5.xml":
            xml_file = os.path.join(
                os.path.dirname(gym.__file__),
                "envs", "mujoco", "assets", "pusher_v5.xml"
            )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # observation space: 7 qpos + 7 qvel + 3 fingertip + 3 object + 3 goal = 23
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64  # 11+11+3+3+3)
        )
    

        # salva le masse originali (come CustomHopper)
        self.original_masses = np.copy(self.model.body_mass[1:])

        # source domain: oggetto più pesante (es. simulare attrito diverso)
        if domain == 'source':
            self.model.body_mass[self._get_object_body_id()] *= 1.5

    # ------------------------------------------------------------------ #
    #  helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_object_body_id(self):
    # body_names non esiste in MuJoCo recente, si usa mj_name2id
        import mujoco
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")

    def _get_obs(self):
        """Stesso schema del CustomHopper: qpos + qvel + xpos."""
        qpos = self.data.qpos.flatten()   # 7 valori
        qvel = self.data.qvel.flatten()   # 7 valori

        # posizioni cartesiane di fingertip, object e goal
        fingertip = self.data.body("tips_arm").xpos.flatten()  # 3
        obj       = self.data.body("object").xpos.flatten()     # 3
        goal      = self.data.body("goal").xpos.flatten()       # 3

        return np.concatenate([qpos, qvel, fingertip, obj, goal])

    # ------------------------------------------------------------------ #
    #  step & reward                                                       #
    # ------------------------------------------------------------------ #

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        reward, reward_info = self._get_rew(obs, action)
        terminated = False  # Pusher non ha condizione di termine
        info = {**reward_info}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def _get_rew(self, obs, action):
        fingertip = obs[22:25]  # ✅ era [14:17]
        obj       = obs[25:28]  # ✅ era [17:20]
        goal      = obs[28:31]  # ✅ era [20:23]

        reward_near = -self._reward_near_weight    * np.linalg.norm(fingertip - obj)
        reward_dist = -self._reward_dist_weight    * np.linalg.norm(obj - goal)
        reward_ctrl = -self._reward_control_weight * np.sum(np.square(action))

        reward = reward_near + reward_dist + reward_ctrl

        reward_info = {
            "reward_near": reward_near,
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
        }
        return reward, reward_info

    # ------------------------------------------------------------------ #
    #  reset                                                               #
    # ------------------------------------------------------------------ #

    def reset_model(self):
        noise = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(-noise, noise,
                                                        size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise,
                                                        size=self.model.nv)
        self.set_state(qpos, qvel)

        if self.custom_domain_randomization:
            self.set_random_parameters()

        return self._get_obs()

    # ------------------------------------------------------------------ #
    #  domain randomization (stesso schema del CustomHopper)              #
    # ------------------------------------------------------------------ #

    def sample_parameters(self):
        low  = self.original_masses[1:] * (1 - self.dr_percentage)
        high = self.original_masses[1:] * (1 + self.dr_percentage)
        return self.np_random.uniform(low, high)

    def set_random_parameters(self):
        self.set_parameters(self.sample_parameters())

    def get_parameters(self):
        return np.array(self.model.body_mass[1:])

    def set_parameters(self, masses):
        """Modifica le masse a runtime — stesso metodo del CustomHopper."""
        self.model.body_mass[2:] = masses


# ------------------------------------------------------------------ #
#  Registrazione environments                                          #
# ------------------------------------------------------------------ #

gym.register(
    id="CustomPusher-v0",
    entry_point=f"{__name__}:CustomPusher",
    max_episode_steps=500,
)

gym.register(
    id="CustomPusher-source-v0",
    entry_point=f"{__name__}:CustomPusher",
    max_episode_steps=500,
    kwargs={"domain": "source"}
)

gym.register(
    id="CustomPusher-UDR-v0",
    entry_point=f"{__name__}:CustomPusher",
    max_episode_steps=500,
    kwargs={"dr_active": True, "domain": "source", "dr_percentage": 0.2}
)