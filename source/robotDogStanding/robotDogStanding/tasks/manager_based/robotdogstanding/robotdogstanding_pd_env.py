from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaacsim.core.simulation_manager import SimulationManager


class robotdogStandingEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = cfg.sim.dt
        self._init_buffers()

    def _init_buffers(self):
        self.default_dof_pos = torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = self.scene["robot"].data.joint_pos[:].clone()

        # Fixed GO2 standing target pose in joint order:
        # FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        self.standing_joint_pos_target = torch.tensor(
            [
                0.0,
                -0.7853403141361257,
                1.2216404886561956,
                0.0,
                -0.7853403141361257,
                1.2216404886561956,
                0.0,
                -0.7853403141361257,
                1.2216404886561956,
                0.0,
                -0.7853403141361257,
                1.2216404886561956,
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Conservative PD gains for standing stabilization.
        self.pd_kp = torch.tensor(
            [40.0, 60.0, 60.0, 40.0, 60.0, 60.0, 40.0, 60.0, 60.0, 40.0, 60.0, 60.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.pd_kd = torch.tensor(
            [5.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 7.0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Tightened safe torque range.
        self.pd_torque_min = -80.0
        self.pd_torque_max = 80.0

    def post_physics_step(self):
        super().post_physics_step()
        self.last_dof_pos[:] = self.scene["robot"].data.joint_pos[:].clone()

    def _sync_action_manager_buffers(self):
        """Keep action-dependent observations/rewards well-defined while bypassing RL control."""
        if not hasattr(self, "action_manager"):
            return

        current_action = getattr(self.action_manager, "action", None)
        prev_action = getattr(self.action_manager, "prev_action", None)

        if current_action is not None and prev_action is not None:
            prev_action.copy_(current_action)
            current_action.zero_()
        elif current_action is not None:
            current_action.zero_()

    def compute_pd_torque(self) -> torch.Tensor:
        robot = self.scene["robot"]
        joint_pos = robot.data.joint_pos[:, :12]
        joint_vel = robot.data.joint_vel[:, :12]

        position_error = self.standing_joint_pos_target.unsqueeze(0) - joint_pos
        velocity_error = -joint_vel
        torque = self.pd_kp.unsqueeze(0) * position_error + self.pd_kd.unsqueeze(0) * velocity_error
        return torch.clamp(torque, min=self.pd_torque_min, max=self.pd_torque_max)

    def _apply_pd_torque(self, torque: torch.Tensor):
        robot = self.scene["robot"]
        if hasattr(robot, "set_joint_effort_target"):
            robot.set_joint_effort_target(torque)
        elif hasattr(robot, "write_joint_effort_to_sim"):
            robot.write_joint_effort_to_sim(torque)
        else:
            raise AttributeError("Robot articulation does not expose a supported joint effort API.")

    def _initialize_pd_joint_state(self, env_ids: torch.Tensor):
        """Initialize reset environments near the PD standing target with zero joint velocity."""
        robot = self.scene["robot"]
        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_vel = robot.data.joint_vel[env_ids].clone()

        joint_pos[:, :12] = self.standing_joint_pos_target.unsqueeze(0)
        joint_vel[:, :12] = 0.0

        if hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        else:
            robot.data.joint_pos[env_ids] = joint_pos
            robot.data.joint_vel[env_ids] = joint_vel

        self.last_dof_pos[env_ids] = joint_pos[:, :12]

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Pure PD control: keep the step signature but ignore incoming RL action.
        del action
        self._sync_action_manager_buffers()
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            pd_torque = self.compute_pd_torque()
            self._apply_pd_torque(pd_torque)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.last_dof_pos[:] = self.scene["robot"].data.joint_pos[:].clone()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self._initialize_pd_joint_state(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        self.recorder_manager.record_pre_reset(env_ids)

        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)
        self._initialize_pd_joint_state(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()

        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        self.recorder_manager.record_post_reset(env_ids)

        self.obs_buf = self.observation_manager.compute()

        feet_indices = torch.tensor([15, 16, 17, 18], dtype=torch.int64, device=self.device)
        self.init_feet_positions = self.scene["robot"].data.body_state_w[:, feet_indices, 0:3].clone()
        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        return self.obs_buf, self.extras
