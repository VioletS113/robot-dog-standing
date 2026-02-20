# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics for evaluating robot dog standing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StandingSuccessMetric:
    """
    Tracks success for robot dog standing task.

    Success is defined as: continuously maintaining for duration_sec:
    - base_height > height_threshold
    - |roll| < angle_threshold
    - |pitch| < angle_threshold

    Supports vectorized environments (num_envs >= 1).
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        height_threshold: float = 0.35,
        angle_threshold: float = 0.3,  # radians (~17 degrees)
        duration_sec: float = 1.0,
        contact_threshold: float | None = None,  # Optional: for future contact-based success
    ):
        """
        Args:
            env: The environment instance
            height_threshold: Minimum base height (meters) to be considered standing
            angle_threshold: Maximum absolute roll/pitch (radians) to be considered upright
            duration_sec: How long conditions must be maintained (seconds)
            contact_threshold: Optional contact force threshold for hind feet (for future use)
        """
        self.env = env
        self.height_threshold = height_threshold
        self.angle_threshold = angle_threshold
        self.duration_sec = duration_sec
        self.contact_threshold = contact_threshold

        # Calculate required number of consecutive steps
        # env.step_dt is the time per step (decimation * sim.dt)
        self.required_steps = int(duration_sec / env.step_dt)

        # Track consecutive success steps for each environment
        self.consecutive_success_steps = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )

        # Track if success was achieved in current episode
        self.success_achieved = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def compute_conditions(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute whether standing conditions are met.

        Returns:
            conditions_met: Boolean tensor (num_envs,) indicating if all conditions are met
            metrics: Dictionary of individual metric values for logging
        """
        robot = self.env.scene["robot"]

        # Get base height
        base_height = robot.data.root_pos_w[:, 2]

        # Get roll and pitch from quaternion
        quat = robot.data.root_quat_w
        roll, pitch, yaw = euler_xyz_from_quat(quat)

        # Check conditions
        height_ok = base_height > self.height_threshold
        roll_ok = torch.abs(roll) < self.angle_threshold
        pitch_ok = torch.abs(pitch) < self.angle_threshold

        # All conditions must be met
        conditions_met = height_ok & roll_ok & pitch_ok

        # Optional: check hind feet contact if threshold is provided
        if self.contact_threshold is not None:
            # This is a placeholder for future contact-based success
            # Would require accessing contact sensor data
            pass

        # Collect metrics for logging
        metrics = {
            "height": base_height,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "height_ok": height_ok.float(),
            "roll_ok": roll_ok.float(),
            "pitch_ok": pitch_ok.float(),
        }

        return conditions_met, metrics

    def update(self, reset_buf: torch.Tensor | None = None) -> torch.Tensor:
        """
        Update success tracking based on current conditions.

        Args:
            reset_buf: Optional tensor indicating which environments are resetting

        Returns:
            success: Boolean tensor (num_envs,) indicating current success status
        """
        conditions_met, _ = self.compute_conditions()

        # Update consecutive success steps
        self.consecutive_success_steps = torch.where(
            conditions_met,
            self.consecutive_success_steps + 1,
            torch.zeros_like(self.consecutive_success_steps)
        )

        # Check if duration requirement is met
        success_now = self.consecutive_success_steps >= self.required_steps

        # Update success achieved flag (once True, stays True until reset)
        self.success_achieved = self.success_achieved | success_now

        # Reset for environments that are done
        if reset_buf is not None:
            reset_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_ids) > 0:
                self.consecutive_success_steps[reset_ids] = 0
                self.success_achieved[reset_ids] = False

        return self.success_achieved

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset success tracking for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)

        self.consecutive_success_steps[env_ids] = 0
        self.success_achieved[env_ids] = False

    def get_success_rate(self) -> float:
        """
        Get overall success rate across all environments.

        Returns:
            Success rate as a float between 0 and 1
        """
        return self.success_achieved.float().mean().item()
