# tasks/pickplace_redblock_rl_env_cfg.py
"""
RL task: Pick and Place Red Block with G1 + Inspire Hands
Based on base_scene_pickplace_redblock (table + cube) but extended for RL training.
"""

import math
import torch
#import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

# Reuse existing scene (table + red block)
from .base_scene_pickplace_redblock import TableRedBlockSceneCfg
from .robot_configs import G1RobotPresets
from unitree_rl_lab.tasks.locomotion import mdp
from .camera_configs import CameraPresets
from isaaclab.sensors import ContactSensorCfg

# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class PickPlaceSceneCfg(TableRedBlockSceneCfg):
    """Scene with table, red block and G1 robot."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
        #self_collisions=True,
    )

    # Add target area (green square) on the table
    # target = SceneEntityCfg(
    #     "target_square",
    # )

    head_camera = CameraPresets.g1_front_camera()
    #left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    #right_wrist_camera = CameraPresets.right_inspire_wrist_camera()

    cube_hand_contacts = ContactSensorCfg(
        # POZOR: nahraď prim path kostky podle tvého souboru.
        # V šablonách je to často něco jako "{ENV_REGEX_NS}/RedCube" nebo "{ENV_REGEX_NS}/Cube"
        prim_path="/World/envs/env_.*/Object",
        update_period=0.0,
        history_length=4,
        debug_vis=True,
        # filtruj JEN kontakty s pravou rukou/prsty (regex podle tvého názvosloví článků)
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/(right_.*|R_.*)"],
    )



# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[
            "right_shoulder.*",
            "right_elbow.*",
            "right_wrist.*",
            "R_.*",
        ],
        scale=1.0, use_default_offset=True
    )


def image_flattened(env, sensor_cfg, data_type="rgb"):
    img = mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type)  # (N,H,W,C)
    return torch.flatten(img, start_dim=1)

# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        concatenate_terms = True
        # Robot joint states
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # Object (cube)
        #cube_position = ObsTerm(func=mdp.object_root_pos, params={"asset_cfg": SceneEntityCfg("object")})
        #cube_velocity = ObsTerm(func=mdp.object_root_vel, params={"asset_cfg": SceneEntityCfg("object")})

        # Target area
        #target_position = ObsTerm(func=mdp.target_root_pos, params={"asset_cfg": SceneEntityCfg("target_square")})

        # Last action (helps policy with dynamics)
        last_action = ObsTerm(func=mdp.last_action)

        head_rgb = ObsTerm(
            func=image_flattened,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb"},
        )
        # right_wrist_rgb = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb"},
        # )
        # left_wrist_rgb = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb"},
        # )

        def __post_init__(self):
            self.concatenate_terms = True   # keep images separate from proprio states
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Distance hand → cube
    reach_cube = RewTerm(
        func=mdp.distance_hand_object,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    # Reward when cube is stably grasped
    grasp_stable = RewTerm(
        func=mdp.is_grasped_stable,
        weight=6.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # self_collision = RewTerm(
    #     func=mdp.self_collision_penalty,
    #     weight=-2.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


    # Encourage moving cube closer to target
    move_to_target = RewTerm(
        func=mdp.distance_object_target,
        weight=-1.0,
        params={"object_cfg": SceneEntityCfg("object"), "target_cfg": SceneEntityCfg("target_square")},
    )

    # Big reward for placing cube inside target
    place = RewTerm(
        func=mdp.is_placed,
        weight=5.0,
        params={"object_cfg": SceneEntityCfg("object"), "target_cfg": SceneEntityCfg("target_square")},
    )

    # Penalize action jitter
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    

    # prevent erratic movement
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.is_placed,
        params={"object_cfg": SceneEntityCfg("object"), "target_cfg": SceneEntityCfg("target_square")},
    )

    cube_fallen = DoneTerm(
        func=mdp.object_below_height,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.6},  # výška stolu ~0.75, dej trochu rezervu
    )

    # Terminations
    # self_collision = DoneTerm(
    #     func=mdp.has_self_collision,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 50.0},
    # )


# -----------------------------------------------------------------------------
# Final Environment Config
# -----------------------------------------------------------------------------
@configclass
class PickPlaceRedBlockRLEnvCfg(ManagerBasedRLEnvCfg):
    """RL Environment for Pick&Place Red Block with G1 + Inspire Hands."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=2, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        print(">>> Scene config entities:", list(self.scene.__dict__.keys()))
        self.decimation = 1
        self.episode_length_s = 1
        self.sim.dt = 0.01
        self.sim.render_interval = 2

@configclass
class PickPlaceRedBlockRLEnvPlayCfg(PickPlaceRedBlockRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4