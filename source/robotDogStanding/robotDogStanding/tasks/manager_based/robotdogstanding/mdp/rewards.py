# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.math import euler_xyz_from_quat
import math
import numpy as np
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply, quat_apply_yaw, quat_conjugate
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
_debug_counter = 0



# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)

def stay_upright(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
    diff = torch.abs(target - z_pos)
    reward = 1 - diff

    # print(env.scene["robot"].data.root_pos_w[:, 2])

    return reward


def balancing_on_four(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return 0.1 * torch.ones(env.num_envs, device=env.device)


# def base_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]
#     terminated = z_pos < 0.10
#     return terminated.bool()

# def standing_pose(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     return

def joint_pose_deviation_l2(env, asset_cfg: SceneEntityCfg, target: dict):
    asset = env.scene[asset_cfg.name]
    joint_names = list(target.keys())
    target_values = torch.tensor([target[name] for name in joint_names], device=env.device)

    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos[:, joint_ids]
    # print(joint_pos)

    error = joint_pos - target_values
    reward = torch.exp(-0.5 * torch.norm(error, dim=-1))
    # print(reward)
    return reward


def base_too_low(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, grace_steps=60):
    z_pos = env.scene[asset_cfg.name].data.root_pos_w[:, 2]

    grace_over = env.episode_length_buf >= grace_steps

    too_low = z_pos < 0.15

    terminated = torch.logical_and(grace_over, too_low)
    # print(z_pos)
    return terminated


def reward_upright_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w
    roll, pitch, _ = euler_xyz_from_quat(quat)

    TARGET_PITCH = -math.pi / 3
    std_dev = 0.20

    reward = torch.exp(-(pitch - TARGET_PITCH) ** 2 / (2 * std_dev ** 2))

    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/upright_pitch", reward.mean().item())
    # print(reward)
    return reward

def stand_upright(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device='cuda:0').repeat(env.num_envs, 1)
    upright_vec = torch.tensor([0.2, 0., 1.0], dtype=torch.float, device='cuda:0').repeat(env.num_envs, 1)
    forward = quat_apply(asset.data.root_quat_w, forward_vec)
    upright_vec = quat_apply_yaw(asset.data.root_quat_w, upright_vec)
    cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
    reward = torch.square(0.5 * cosine_dist + 0.5)
    return reward

def upright_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    quat = env.scene[asset_cfg.name].data.root_quat_w
    roll, pitch, _ = euler_xyz_from_quat(quat)

    return -torch.square(pitch) - torch.square(roll)


def reward_feet_air_time_simple(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene["contact_forces"]  # make sure this is the correct name

    # Hind foot names
    foot_names = ["RL_foot", "RR_foot"]
    foot_ids = [asset.body_names.index(name) for name in foot_names]

    # Get vertical velocities
    foot_vel_z = asset.data.body_vel_w[:, foot_ids, 2]

    # Get contact forces (net contact forces in world frame)
    contact_z = contact_sensor.data.net_forces_w[:, foot_ids, 2]  # ✅ corrected line
    contact = contact_z > 1.0

    impact = (foot_vel_z < -0.5) & contact
    reward = torch.sum(impact.float(), dim=1) * 0.5

    # 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/feet_air_time_simple", reward.mean().item())
    return reward


def reward_lift_up_linear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    reward = (root_height - 0.15) / (0.52 - 0.15)
    reward = torch.clamp(reward, 0., 1.)
    return reward


def reward_foot_shift(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalizes early foot movement after reset to encourage stable, planted feet.
    Higher reward when feet stay near initial positions. Only applies during early steps.
    """
    asset = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w  # shape: (num_envs, num_bodies, 3)

    # Get foot indices
    front_ids = [asset.body_names.index(name) for name in ["FL_foot", "FR_foot"]]
    rear_ids = [asset.body_names.index(name) for name in ["RL_foot", "RR_foot"]]

    # Save initial foot positions if not already done
    if getattr(env, "_init_foot_pos", None) is None:
        env._init_foot_pos = torch.clone(foot_pos)

    init_foot_pos = env._init_foot_pos

    # Compute per-foot displacement in XY plane
    # front_shift = torch.norm(foot_pos[:, front_ids, :2] - init_foot_pos[:, front_ids, :2], dim=-1).mean(dim=1)
    rear_shift = torch.norm(foot_pos[:, rear_ids, :2] - init_foot_pos[:, rear_ids, :2], dim=-1).mean(dim=1)

    # Only apply during early steps
    grace_steps = 50
    condition = (env.episode_length_buf < grace_steps).float()

    # Penalize total shift
    reward = -(rear_shift) * condition

    # 👇 Add this to log to TensorBoard (if enabled)
    if hasattr(env, "logger"):
        env.logger.log_scalar("reward/reward_foot_shift", reward.mean().item())
    print(reward)
    return reward


# def reward_rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """
#     Encourages rear feet to stay in contact and penalizes rearing behavior
#     where the rear feet are in the air but the calves are contacting the ground.

#     Aims to catch the undesirable situation of the robot tipping backward.
#     """
#     contact_forces = env.scene[sensor_cfg.name].data.net_forces_w  # (num_envs, num_bodies, 3)

#     # Index rear feet and rear calves
#     rear_foot_names = ["RL_foot", "RR_foot"]
#     rear_calf_names = ["RL_calf", "RR_calf"]
#     body_names = sensor_cfg.body_names if sensor_cfg.body_names else env.scene[sensor_cfg.name].body_names

#     rear_foot_ids = [body_names.index(name) for name in rear_foot_names]
#     rear_calf_ids = [body_names.index(name) for name in rear_calf_names]

#     # Contact = z-force < threshold => not in contact
#     foot_in_air = contact_forces[:, rear_foot_ids, 2] < 1.0   # shape (num_envs, 2)
#     calf_in_contact = contact_forces[:, rear_calf_ids, 2] >= 1.0

#     # Unhealthy: calves touching ground, but feet not
#     unhealthy = torch.logical_and(calf_in_contact, foot_in_air)

#     # Reward is:
#     # +1 if both feet are in contact
#     # +penalty if unhealthy case is triggered
#     feet_contact = torch.all(~foot_in_air, dim=1).float()
#     unhealthy_penalty = unhealthy.sum(dim=1).float()  # 0 to 2

#     # grace_steps = 50
#     # active = (env.episode_length_buf >= grace_steps).float()
#     # reward = (feet_contact - unhealthy_penalty) * active
#     reward = (feet_contact - unhealthy_penalty)

#     return reward
def reward_rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Encourages rear feet to stay in contact and penalizes rearing behavior
    where the rear feet are in the air but the rear legs (hips, thighs, calves) are contacting the ground.

    Aims to catch the undesirable situation of the robot tipping backward.
    """
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w  # (num_envs, num_bodies, 3)

    # Index rear feet and rear leg segments
    rear_foot_names = ["RL_foot", "RR_foot"]
    rear_leg_names = ["RL_calf", "RR_calf", "RL_thigh", "RR_thigh", "RL_hip", "RR_hip"]
    body_names = sensor_cfg.body_names if sensor_cfg.body_names else env.scene[sensor_cfg.name].body_names

    rear_foot_ids = [body_names.index(name) for name in rear_foot_names]
    rear_leg_ids = [body_names.index(name) for name in rear_leg_names]

    # Determine contact states
    # foot_touch_ground = contact_forces[:, rear_foot_ids, 2] # shape (num_envs, 2)

    raw_force = torch.sum(contact_forces[:, rear_foot_ids, 2])
    max_expected_force = 100.0  # Newtons, or pick based on your robot
    foot_touch_ground = torch.clamp(raw_force / max_expected_force, 0.0, 1.0)

    leg_in_contact = contact_forces[:, rear_leg_ids, 2] >= 1.0  # shape (num_envs, 6)

    # Unhealthy if any leg segment is touching while corresponding foot is off the ground
    # For simplicity, treat all leg contacts as penalizing regardless of alignment
    # unhealthy = torch.any(leg_in_contact, dim=1) & torch.any(foot_in_air, dim=1)

    # Compute reward
    # feet_contact = torch.all(foot_touch_ground, dim=1).float()
    # unhealthy_penalty = unhealthy.float()

    # grace_steps = 60
    # active = (env.episode_length_buf >= grace_steps).float()
    # reward = (feet_contact - (unhealthy_penalty * 1.5)) * active
    # i added a *1.5 just to make the penalty weigh more.
    print(foot_touch_ground)

    return foot_touch_ground


def feet_clearance_cmd_linear(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    phases = 1 - torch.abs(1.0 - torch.clip((env.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    feet_indices = torch.tensor([15,16,17,18], dtype=torch.int64, device='cuda:0')
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    foot_height = (foot_positions[:, -2:, 2]).view(env.num_envs, -1)# - reference_heights
    terrain_at_foot_height = env._get_heights_at_points(foot_positions[:, -2:, :2])
    target_height = 0.05 * phases + terrain_at_foot_height + 0.02
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - env.desired_contact_states[:, -2:])
    condition = env.episode_length_buf > 30
    rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
    rew_foot_clearance = rew_foot_clearance
    return torch.sum(rew_foot_clearance, dim=1)


def action_rate_l2_early_training(env) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment

    reward = torch.sum(is_contact, dim=1)
    cond = env.episode_length_buf > 30
    reward = reward * cond.float()
    return reward

def feet_slip(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    foot_velocities = asset.data.body_state_w[:, feet_indices, 7:10]  # shape: (num_envs, num_bodies, 13)
    foot_velocities_ang = asset.data.body_state_w[:, feet_indices, 10:13]
    condition = foot_positions[:, :, 2] < 0.03
    # xy lin vel
    foot_velocities = torch.square(torch.norm(foot_velocities[:, :, 0:2], dim=2).view(env.num_envs, -1))
    # yaw ang vel
    foot_ang_velocities = torch.square(torch.norm(foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(env.num_envs, -1))
    rew_slip = torch.sum(condition.float() * (foot_velocities + foot_ang_velocities), dim=1)
    return rew_slip




def applied_torque_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # # TODO: We need to fix this to support implicit joints.
    # condition = env.episode_length_buf <= 30
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    torque_limits_penalty = torch.clamp(torch.sum(out_of_limits, dim=1), max=10.0)
    return torque_limits_penalty


def foot_shift(env,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
    desired_foot_positions = torch.clone(env.init_feet_positions[:, 2:])
    desired_foot_positions[:, :, 2] = 0.02
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    rear_foot_shift = torch.norm(foot_positions[:, 2:] - desired_foot_positions, dim=-1).mean(dim=1)
    init_ffoot_positions = torch.clone(env.init_feet_positions[:, :2])
    front_foot_shift = torch.norm( torch.stack([
            (init_ffoot_positions[:, :, 0] - foot_positions[:, :2, 0]).clamp(min=0),
            torch.abs(init_ffoot_positions[:, :, 1] - foot_positions[:, :2, 1])
        ], dim=-1), dim=-1).mean(dim=1)
    condition = env.episode_length_buf < 30
    reward = (front_foot_shift + rear_foot_shift) * condition.float()
    return reward

def low_thigh_contacts(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    body_state = env.scene["robot"].data.body_state_w  # shape: (num_envs, num_bodies, 13)
    thigh_names = ["RL_thigh", "RR_thigh"]
    body_names = env.scene["robot"].body_names
    thigh_ids = [body_names.index(name) for name in thigh_names]

    head_names = ["Head_upper", "Head_lower"]
    head_ids = [body_names.index(name) for name in head_names]

    thigh_z = body_state[:, thigh_ids, 2]  # shape: (num_envs, 4)
    head_z = body_state[:, head_ids, 2]
    penalty = torch.sum(torch.clamp(0.05 - thigh_z, min=0.0), dim=1)
    head_z_vec = torch.tensor([1.], dtype=torch.float, device='cuda:0').repeat(env.num_envs)
    thigh_z_vec = torch.tensor([0.78], dtype=torch.float, device='cuda:0').repeat(env.num_envs)
    reward = torch.sum(torch.clamp(thigh_z, min=0.0), dim=1) + torch.sum(torch.clamp(head_z, min=0.0), dim=1) - head_z_vec - thigh_z_vec
    return reward
    # return penalty


def upward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    # reward = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward
    # upward_error = torch.square(asset.data.projected_gravity_b[:, 2] - (-1))
    # return torch.exp(-upward_error / std**2)


def feet_distance_y_exp(
        env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, [17,18], :] - asset.data.root_link_pos_w[
                                                                                      :, :
                                                                                      ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for i in range(2):
        footsteps_in_body_frame[:, i, :] = quat_apply(
            quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std ** 2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def vertical_alignment(env:ManagerBasedRLEnv,  std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"))-> torch.Tensor:
    hip_y = env.scene["robot"].data.body_pos_w[:, [4,5], 1]
    foot_y = env.scene["robot"].data.body_pos_w[:, [17,18], 1]
    vertical_offset = torch.abs(hip_y - foot_y)
    reward = torch.exp(-torch.sum(vertical_offset, dim=1) / std ** 2)
    return reward


def action_q_diff(env:ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    condition = env.episode_length_buf <= 30
    actions = torch.clip(env.action_manager.action, -100, 100)
    q_diff_buf = torch.abs(asset.data.default_joint_pos + 0.25 * actions - asset.data.joint_pos)
    reward = torch.sum(torch.square(q_diff_buf), dim=-1) * condition.float()
    return reward