from __future__ import annotations

import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from bdx.tasks.utils import MotionLoader
from bdx.tasks.utils.mti3_imu import Mti3ImuModel, create_mti3_imu_sensor_prims, create_mti3_lab_imu_sensor

from .bdx_amp_env_cfg import BdxAmpEnvCfg


class BdxAmpEnv(DirectRLEnv):
    cfg: BdxAmpEnvCfg

    def __init__(self, cfg: BdxAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        motion_files = [
            os.path.join(self.cfg.motion_folder_path, f)
            for f in os.listdir(self.cfg.motion_folder_path)
            if f.endswith(".json")
        ]
        self.motion_loader = MotionLoader(
            "bdx",
            motion_files,
            self.step_dt,
            self.device,
            joint_names=self.robot.data.joint_names,
        )

        joint_pos_limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(self.robot.data, "joint_pos_limits", None)
        self._joint_pos_lower = joint_pos_limits[..., 0] if joint_pos_limits is not None else None
        self._joint_pos_upper = joint_pos_limits[..., 1] if joint_pos_limits is not None else None

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.left_toe_body_index = self.robot.data.body_names.index("left_foot")
        self.right_toe_body_index = self.robot.data.body_names.index("right_foot")
        self.trunk_body_index = self.robot.data.body_names.index("trunk")
        self.imu_body_index = self.robot.data.body_names.index(self.cfg.imu_body)

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )
        self.commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_scale = torch.tensor(
            [
                self.cfg.command_linear_velocity_scale,
                self.cfg.command_linear_velocity_scale,
                self.cfg.command_angular_velocity_scale,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float32, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.average_velocity_steps = max(1, int(round(self.cfg.average_velocity_window_s / self.step_dt)))
        self.velocity_history = torch.zeros(
            (self.num_envs, self.average_velocity_steps, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self.mti3_imu = Mti3ImuModel(
            self.cfg,
            self.num_envs,
            self.device,
            self.step_dt,
            getattr(self, "_imu_sensor_paths", []),
            getattr(self, "_lab_imu_sensor", None),
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self.robot
        self._lab_imu_sensor = create_mti3_lab_imu_sensor(self.cfg)
        if self._lab_imu_sensor is not None:
            self.scene.sensors[self.cfg.imu_sensor_name] = self._lab_imu_sensor
        self._imu_sensor_paths = create_mti3_imu_sensor_prims(self.cfg, self.scene.cfg.num_envs)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone()

    def _apply_action(self):
        actions = torch.clamp(self.actions, -1.0, 1.0)
        target = self.robot.data.default_joint_pos + self.cfg.action_scale * actions
        if self._joint_pos_lower is not None:
            target = torch.clamp(target, self._joint_pos_lower, self._joint_pos_upper)

        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        root_position = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_orientation_quat_wxyz = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_orientation_quat_xyzw = root_orientation_quat_wxyz[:, [1, 2, 3, 0]]
        world_linear_velocity = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        world_angular_velocity = self.robot.data.body_ang_vel_w[:, self.ref_body_index]

        policy_orientation_quat_xyzw = root_orientation_quat_xyzw
        policy_angular_velocity = world_angular_velocity
        if self.cfg.use_isaac_imu_sensor:
            policy_orientation_quat_xyzw, policy_angular_velocity = self.mti3_imu.read(
                self.robot.data,
                self.imu_body_index,
            )

        left_toe_position = self.robot.data.body_pos_w[:, self.left_toe_body_index]
        right_toe_position = self.robot.data.body_pos_w[:, self.right_toe_body_index]
        toe_positions = torch.stack((left_toe_position, right_toe_position), dim=1)
        amp_local_toe_positions = self.motion_loader.localize_positions(
            root_position,
            root_orientation_quat_xyzw,
            toe_positions,
        )
        policy_local_toe_positions = self.motion_loader.localize_positions(
            root_position,
            policy_orientation_quat_xyzw,
            toe_positions,
        )

        (
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            world_linear_velocity,
            world_angular_velocity,
        )
        (
            policy_orientation_quat_normalized_heading,
            _,
            policy_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            policy_orientation_quat_xyzw,
            torch.zeros_like(world_linear_velocity),
            policy_angular_velocity,
        )

        joints_positions_vectorized = self.motion_loader.vectorize_joint_positions(self.robot.data.joint_pos)
        amp_obs = compute_obs(
            root_position[:, 2:3],
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            self.robot.data.joint_vel,
            amp_local_toe_positions[:, 0],
            amp_local_toe_positions[:, 1],
        )
        policy_state_obs = compute_policy_obs(
            policy_orientation_quat_normalized_heading,
            policy_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            self.robot.data.joint_vel,
            policy_local_toe_positions[:, 0],
            policy_local_toe_positions[:, 1],
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        policy_obs = torch.cat((policy_state_obs, self._get_command_observations()), dim=-1)

        return {"policy": policy_obs}

    def _get_rewards(self) -> torch.Tensor:
        root_orientation_quat_wxyz = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_orientation_quat_xyzw = root_orientation_quat_wxyz[:, [1, 2, 3, 0]]
        world_linear_velocity = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        world_angular_velocity = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        if self.cfg.use_average_velocities:
            average_root_velocity = self._update_average_root_velocity(world_linear_velocity, world_angular_velocity)
            reward_linear_velocity = average_root_velocity[:, :3]
            reward_angular_velocity = average_root_velocity[:, 3:]
        else:
            reward_linear_velocity = world_linear_velocity
            reward_angular_velocity = world_angular_velocity

        (
            _,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            reward_linear_velocity,
            reward_angular_velocity,
        )

        linear_velocity_error = torch.sum(
            torch.square(self.commands[:, :2] - root_linear_velocity_normalized_heading[:, :2]),
            dim=-1,
        )
        angular_velocity_error = torch.square(
            self.commands[:, 2] - local_root_angular_velocity_normalized_heading[:, 2]
        )
        linear_velocity_reward = (
            torch.exp(-linear_velocity_error / self.cfg.command_tracking_sigma)
            * self.cfg.linear_velocity_xy_reward_scale
        )
        angular_velocity_reward = (
            torch.exp(-angular_velocity_error / self.cfg.command_tracking_sigma)
            * self.cfg.angular_velocity_z_reward_scale
        )
        action_rate_reward = (
            torch.sum(torch.square(self.previous_actions - self.actions), dim=-1)
            * self.cfg.action_rate_reward_scale
            * self.step_dt
        )
        torque_reward = (
            torch.sum(torch.square(self.robot.data.applied_torque), dim=-1)
            * self.cfg.torque_reward_scale
            * self.step_dt
        )
        stand_still_reward = (
            torch.sum(torch.abs(self.robot.data.joint_pos - self.robot.data.default_joint_pos), dim=-1)
            * (torch.linalg.norm(self.commands, dim=-1) < self.cfg.stand_still_command_threshold)
            * self.cfg.stand_still_reward_scale
        )
        reward = (
            linear_velocity_reward
            + angular_velocity_reward
            + torque_reward
            + action_rate_reward
            + stand_still_reward
        )
        return torch.clamp(reward, min=0.0)

    def _update_average_root_velocity(
        self,
        world_linear_velocity: torch.Tensor,
        world_angular_velocity: torch.Tensor,
    ) -> torch.Tensor:
        root_velocity = torch.cat((world_linear_velocity, world_angular_velocity), dim=-1)
        if self.average_velocity_steps == 1:
            return root_velocity

        self.velocity_history[:, :-1] = self.velocity_history[:, 1:].clone()
        self.velocity_history[:, -1] = root_velocity
        return torch.mean(self.velocity_history, dim=1)

    def _sample_commands(self, env_ids: torch.Tensor, motion_ids: np.ndarray | None = None):
        if self.cfg.sample_commands_from_reference:
            if motion_ids is None:
                motion_ids = self.motion_loader.sample_motion_ids(len(env_ids))
            commands = self.motion_loader.get_motion_commands(
                motion_ids,
                use_average=self.cfg.use_average_velocities,
            )
            lower = torch.tensor(
                [
                    self.cfg.command_linear_x_range[0],
                    self.cfg.command_linear_y_range[0],
                    self.cfg.command_yaw_range[0],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            upper = torch.tensor(
                [
                    self.cfg.command_linear_x_range[1],
                    self.cfg.command_linear_y_range[1],
                    self.cfg.command_yaw_range[1],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self.commands[env_ids] = torch.clamp(commands, lower, upper)
            return

        self.commands[env_ids, 0] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_linear_x_range[0],
            self.cfg.command_linear_x_range[1],
        )
        self.commands[env_ids, 1] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_linear_y_range[0],
            self.cfg.command_linear_y_range[1],
        )
        self.commands[env_ids, 2] = torch.empty(len(env_ids), dtype=torch.float32, device=self.device).uniform_(
            self.cfg.command_yaw_range[0],
            self.cfg.command_yaw_range[1],
        )

    def _get_command_observations(self) -> torch.Tensor:
        return self.commands * self.command_scale

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        trunk_z = self.robot.data.body_pos_w[:, self.trunk_body_index, 2]

        if self.cfg.early_termination:
            died = trunk_z < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _get_robot_reset_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        reset_device = getattr(self.robot.instantaneous_wrench_composer, "device", self.device)
        try:
            return env_ids.to(device=torch.device(str(reset_device)))
        except (TypeError, RuntimeError):
            return env_ids

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reset_env_ids = self._get_robot_reset_env_ids(env_ids)
        self.robot.reset(reset_env_ids)
        super()._reset_idx(reset_env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.reset_strategy.startswith("random"):
            random_root_state, random_joint_pos, random_joint_vel, motion_ids, times = self._sample_reference_state(
                env_ids, start="start" in self.cfg.reset_strategy
            )
            root_state = random_root_state
            joint_pos = random_joint_pos
            joint_vel = random_joint_vel
        else:
            motion_ids = None
            times = np.zeros(len(env_ids), dtype=np.float64)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.velocity_history[env_ids] = root_state[:, 7:].unsqueeze(1).expand(-1, self.average_velocity_steps, -1)
        self.mti3_imu.reset(env_ids, root_state[:, 3:7][:, [1, 2, 3, 0]], root_state[:, 10:13])
        self._sample_commands(env_ids, motion_ids)

        amp_observations = self.collect_reference_motions(len(env_ids), times, motion_ids)
        self.amp_observation_buffer[env_ids] = amp_observations.view(len(env_ids), self.cfg.num_amp_observations, -1)

    def _sample_reference_state(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        num_samples = len(env_ids)
        motion_ids = self.motion_loader.sample_motion_ids(num_samples)
        times = np.zeros(num_samples, dtype=np.float64) if start else self.motion_loader.sample_times(motion_ids)
        (
            root_position,
            root_orientation_quat_xyzw,
            root_linear_velocity,
            root_angular_velocity,
            joint_pos,
            joint_vel,
            _,
            _,
        ) = self.motion_loader.sample(num_samples, motion_ids=motion_ids, times=times, return_full_state=True)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = root_position + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_orientation_quat_xyzw[:, [3, 0, 1, 2]]
        root_state[:, 7:10] = root_linear_velocity
        root_state[:, 10:13] = root_angular_velocity

        return root_state, joint_pos, joint_vel, motion_ids, times

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | None = None, motion_ids: np.ndarray | None = None
    ) -> torch.Tensor:
        if motion_ids is None:
            motion_ids = self.motion_loader.sample_motion_ids(num_samples)

        if current_times is None:
            current_times = self.motion_loader.sample_times(motion_ids)

        history_times = (
            np.expand_dims(current_times, axis=-1) - self.step_dt * np.arange(0, self.cfg.num_amp_observations)
        ).reshape(-1)
        history_motion_ids = np.repeat(motion_ids, self.cfg.num_amp_observations)

        amp_observation = compute_obs(
            *self.motion_loader.sample(
                history_motion_ids.shape[0],
                motion_ids=history_motion_ids,
                times=history_times,
            )
        )
        return amp_observation.view(num_samples, -1)


@torch.jit.script
def compute_obs(
    root_height: torch.Tensor,
    root_orientation_quat_normalized_heading: torch.Tensor,
    root_linear_velocity_normalized_heading: torch.Tensor,
    local_root_angular_velocity_normalized_heading: torch.Tensor,
    joints_positions_vectorized: torch.Tensor,
    joints_velocities: torch.Tensor,
    left_toe_position: torch.Tensor,
    right_toe_position: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            root_height,
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
        ),
        dim=-1,
    )


@torch.jit.script
def compute_policy_obs(
    root_orientation_quat_normalized_heading: torch.Tensor,
    local_root_angular_velocity_normalized_heading: torch.Tensor,
    joints_positions_vectorized: torch.Tensor,
    joints_velocities: torch.Tensor,
    left_toe_position: torch.Tensor,
    right_toe_position: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            root_orientation_quat_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
        ),
        dim=-1,
    )
