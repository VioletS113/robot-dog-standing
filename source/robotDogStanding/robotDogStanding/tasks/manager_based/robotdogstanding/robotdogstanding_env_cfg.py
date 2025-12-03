# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns


from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

##
# Scene definition
##

def base_quat(obs_manager, scene, env_ids):
    return scene["robot"].data.root_quat_wxyz[env_ids]

@configclass
class RobotdogstandingSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    scale = 20,
    # scale={".*_hip_joint": 10.0, "^(?!.*_hip_joint).*": 80.0, },
    clip={".*": (-100.0, 100.0)}
    )


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""
#
#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel,scale=2.0)
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel,scale=0.25)
#
#
#         # observation terms (order preserved)
#         joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel,scale=1.0)
#         joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,scale=0.05)
#
#         projected_gravity = ObsTerm(func=mdp.projected_gravity)
#
#         actions = ObsTerm(func=mdp.last_action)
#
#
#         def __post_init__(self) -> None:
#             self.enable_corruption = False
#             self.concatenate_terms = True
#
#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)

        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel,scale=0.05)
        #base_orientation = ObsTerm(func=base_quat)
#         base_orientation = ObsTerm(
#         func=base_quat,
#         params={"scene": "scene", "env_ids": "env_ids"},
# )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[
    #             "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #             "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #             "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #             "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #         ]),
    #         "position_range": (0.1, 0.3),
    #         "velocity_range": (-0.1 * math.pi, 0.1 * math.pi),
    #     },
    # )
    # #
    # reset_robot_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode='reset',
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[
    #             "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #             "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #             "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #             "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #         ]),
    #         # "pose_range": {
    #         #     "x": (-0.05, 0.05),
    #         #     "y": (-0.05, 0.05),
    #         #     "z": (0.05, 0.1),  # sitting height
    #         #     "roll": (0.0, 0.0),
    #         #     "pitch": (-0.5, -0.3),  # leaning back slightly
    #         #     "yaw": (-0.1, 0.1)
    #         # },
    #         "velocity_range": {
    #             "x": (0.01, 0.01),
    #             "y": (0.01, 0.01),
    #             "z": (0.01, 0.01),
    #             "roll": (0.01, 0.01),
    #             "pitch": (0.01, 0.01),
    #             "yaw": (0.01, 0.01),
    #         },
    #
    #             "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
    #                 "velocity_range": {
    #                     "x": (-0.1, 0.1),
    #                     "y": (-0.1, 0.1),
    #                     "z": (-0.1, 0.15),
    #                     "roll": (-0.1, 0.1),
    #                     "pitch": (-0.8, 0),
    #                     "yaw": (-0.1, 0.1),
    #                 }
    #     }
    # )

    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        # params = {
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=[
        #             "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        #             "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        #             "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        #             "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        #         ])
        # }
    )
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1)
    # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-1)


    #
    # upright_pitch_reward = RewTerm(
    #     func=mdp.reward_upright_pitch,
    #     weight=10.0,  # adjust based on importance
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    stand_upright = RewTerm(
        func=mdp.stand_upright,
        weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # upright_penalty = RewTerm(
    #     func=mdp.upright_penalty,
    #     weight = -5,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )


    reward_lift_up_linear = RewTerm(
        func=mdp.reward_lift_up_linear,
        weight=1,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


    # rear_air_reward = RewTerm(
    #     func=mdp.reward_rear_air,
    #     weight=1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
    #         "RL_foot", "RR_foot", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh", "RL_hip", "RR_hip"
    #     ])}
    # )
    # gait = RewTerm(
    #     func=mdp.reward_feet_clearance,
    #     weight=1,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_early_training,
        weight = -0.03
    )
    smooth_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001
    )

    smooth_acc= RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-07
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10
    )
    feet_clearance_cmd_linear = RewTerm(
        func=mdp.feet_clearance_cmd_linear,
        weight=-300
    )
    #
    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.4
    )
    #
    # applied_torque_limits = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=-0.01
    # )
    #
    foot_shift = RewTerm(
        func=mdp.foot_shift,
        weight=-50
    )
    #
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    #
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!RL_foot$|RR_foot$).*"), "threshold": 0.1},
    )

    # feet_distance_y_exp= RewTerm(
    #     func=mdp.feet_distance_y_exp,
    #     weight=2,
    #     params={"stance_width":0.5, "std":0.2}
    # )




    action_q_diff = RewTerm(
        func=mdp.action_q_diff,
        weight = -1.0,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


    # fell_over = DoneTerm(
    #     func=mdp.base_too_low,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # bad_foot_contacts = DoneTerm(
    #     func=mdp.bad_foot_contacts,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    position_protect = DoneTerm(
        func=mdp.position_protect,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    stand_air_condition = DoneTerm(
        func=mdp.stand_air_condition,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    abrupt_change_condition = DoneTerm(
        func=mdp.abrupt_change_condition,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

##
# Environment configuration
##


@configclass
class RobotdogstandingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RobotdogstandingSceneCfg = RobotdogstandingSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        # modified one original is 5
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


