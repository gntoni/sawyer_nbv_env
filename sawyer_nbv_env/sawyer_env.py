
import robosuite as rs
from robosuite.controllers import load_controller_config
from gym.envs.registration import register
from sawyer_nbv_env.wrappers import MergeObsWrapper, GymWrapper


def CapDataEnv():
    env = GymWrapper(
        MergeObsWrapper(
            rs.make(
                "CaptureDataset",
                robots=["Sawyer"],
                controller_configs=load_controller_config(default_controller="OSC_POSITION"),            
                use_camera_obs=True,  #  use pixel observations
                has_offscreen_renderer=True,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
                camera_names= ["robot0_eye_in_hand"],
                camera_heights=[84],
                camera_widths=[84],
                camera_depths=[True],
            ),
            img_keys = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_depth"],
            vect_keys = ["robot0_eef_pos","robot0_gripper_qpos"]
        ),
        keys=["merged_obs"]
    )
    return env


def liftEnv():
    env = GymWrapper(
        MergeObsWrapper(
            rs.make(
                "LiftNBVCube",
                robots=["Sawyer"],
                controller_configs=load_controller_config(default_controller="OSC_POSITION"),            
                use_camera_obs=True,  #  use pixel observations
                has_offscreen_renderer=True,  # not needed since not using pixel obs
                has_renderer=False,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
                camera_names= ["robot0_eye_in_hand"],
                camera_heights=[84],
                camera_widths=[84],
                camera_depths=[True],
            ),
            img_keys = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_depth"],
            vect_keys = ["robot0_eef_pos","robot0_gripper_qpos"]
        ),
        keys=["merged_obs"]
    )
    return env

def liftEnvPlay():
    env = GymWrapper(
        MergeObsWrapper(        
            rs.make(
                "LiftNBVCube",
                robots=["Sawyer"],
                controller_configs=load_controller_config(default_controller="OSC_POSITION"),
                use_camera_obs=False,  # do not use pixel observations
                has_offscreen_renderer=False,  # not needed since not using pixel obs
                has_renderer=True,  # make sure we can render to the screen
                reward_shaping=True,  # use dense rewards
                control_freq=10,  # control should happen fast enough so that simulation looks smooth
                camera_heights=84,
                camera_widths=84,
                camera_depths=True,            
            ),
            img_keys = ["agentview_image", "agentview_depth"],
            vect_keys = ["robot0_robot-state"]            
        ),
        keys=["merged_obs"]        
    )
    return env

register(
    'CaptureDataset-v0',
    entry_point='sawyer_nbv_env.sawyer_env:CapDataEnv',
    max_episode_steps=500
)

register(
    'SawyerLift-v0',
    entry_point='sawyer_nbv_env.sawyer_env:liftEnv',
    max_episode_steps=500
)

register(
    'SawyerLiftPlay-v0',
    entry_point='sawyer_nbv_env.sawyer_env:liftEnvPlay',
    max_episode_steps=300
)
