
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



def bad_foot_contacts(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene["contact_forces"]
    termination_contact_names =['base', 'Head_upper', 'Head_lower', 'FR_thigh', 'FL_thigh', 'FR_calf', 'FL_calf', 'FR_foot', 'FL_foot', 'RL_thigh', 'RR_thigh']
    allow_initial_contact_names = ['FL_foot', 'FR_foot','RL_foot','RR_foot','RL_calf', 'RR_calf']
    contact_ids = [asset.body_names.index(name) for name in termination_contact_names]
    allow_initial_contact_ids = [asset.body_names.index(name) for name in allow_initial_contact_names]
    return torch.logical_and(
    torch.any(
        torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, contact_ids], dim=-1), dim=1)[0] > 1, dim=1
    ),
    torch.logical_not(torch.logical_and(
        torch.any(torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, allow_initial_contact_ids], dim=-1), dim=1)[0] > 1, dim=1),
        env.episode_length_buf <= 30
    ))
    )

def position_protect( env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    position_protect = torch.logical_and(
        env.episode_length_buf > 3, torch.any(torch.logical_or(
            asset.data.joint_pos< asset.data.joint_pos_limits[0][:,0] + 5 / 180 * np.pi,
            asset.data.joint_pos > asset.data.joint_pos_limits[0][:,1] - 5 / 180 * np.pi
        ), dim=-1))

    return position_protect

def stand_air_condition( env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device='cuda:0')
    foot_positions = asset.data.body_state_w[:, feet_indices, 0:3]
    stand_air_condition = torch.logical_and(
            torch.logical_and(env.episode_length_buf > 3, env.episode_length_buf <= 30),
            torch.any(foot_positions[:, -2:, 2] > 0.06, dim=-1)
        )
    return stand_air_condition

def abrupt_change_condition(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    abrupt_change_condition = torch.logical_and(
            torch.logical_and(env.episode_length_buf > 3, env.episode_length_buf <= 100),
                # torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= 100),
            torch.any(torch.abs(asset.data.joint_pos - env.last_dof_pos) > 0.5, dim=-1)
    )
    return abrupt_change_condition