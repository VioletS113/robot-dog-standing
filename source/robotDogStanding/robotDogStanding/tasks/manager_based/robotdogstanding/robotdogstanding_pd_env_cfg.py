# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class RobotdogstandingPDSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the PD standing task."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Minimal action manager definition kept for ManagerBasedRLEnv compatibility.

    Note:
        The PD environment ignores incoming policy actions and computes torques internally.
    """

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        scale={".*_hip_joint": 1.0, "^(?!.*_hip_joint).*": 1.0},
        clip={".*": (-80.0, 80.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for PD standing."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, scale=1.0)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Event configuration for PD standing."""

    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Conservative rewards for PD debugging/standing behavior."""

    stand_upright = RewTerm(
        func=mdp.stand_upright,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lift_up_linear = RewTerm(
        func=mdp.reward_lift_up_linear,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # position_protect disabled for conservative PD debugging pass to reduce early 4-step failures.
    # position_protect = DoneTerm(
    #     func=mdp.position_protect,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )


@configclass
class RobotdogstandingPDEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL config used by the PD standing environment implementation."""

    scene: RobotdogstandingPDSceneCfg = RobotdogstandingPDSceneCfg(num_envs=16, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
