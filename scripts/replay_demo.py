# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded motion clip on the BDX robot."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


TASK_NAME = "Template-Bdx-Amp-Direct-v0"
NUM_ENVS = 1

parser = argparse.ArgumentParser(description="Replay a recorded BDX demo motion.")
parser.add_argument("--robot", type=str, default="bdx", choices=["bdx"], help="Robot motion set to replay.")
parser.add_argument("--index", type=int, default=0, help="Motion JSON index to replay, e.g. 0 for 0.json.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the selected motion.")
args_cli = parser.parse_args()

app_launcher = AppLauncher()
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import bdx.tasks  # noqa: F401
from bdx.robots import ISAACLAB_ASSETS_DIR


def _resolve_motion_path(robot: str, index: int) -> Path:
    motion_path = Path(ISAACLAB_ASSETS_DIR) / "motions" / robot / f"{index}.json"
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    return motion_path


def _configure_replay_env(motion_folder_path: str):
    env_cfg = parse_env_cfg(
        TASK_NAME,
        num_envs=NUM_ENVS,
        use_fabric=True,
    )
    env_cfg.motion_folder_path = motion_folder_path
    env_cfg.reset_strategy = "default"
    env_cfg.early_termination = False
    env_cfg.use_isaac_imu_sensor = False
    env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
    return env_cfg


def _write_motion_state(replay_env, motion_time: float, env_ids: torch.Tensor):
    motion_ids = np.zeros(replay_env.num_envs, dtype=np.int64)
    motion_times = np.full(replay_env.num_envs, motion_time, dtype=np.float64)
    (
        root_position,
        root_orientation_quat_xyzw,
        root_linear_velocity,
        root_angular_velocity,
        joint_pos,
        joint_vel,
        _,
        _,
    ) = replay_env.motion_loader.sample(
        replay_env.num_envs,
        motion_ids=motion_ids,
        times=motion_times,
        return_full_state=True,
    )

    root_state = replay_env.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = root_position + replay_env.scene.env_origins[env_ids]
    root_state[:, 3:7] = root_orientation_quat_xyzw[:, [3, 0, 1, 2]]
    root_state[:, 7:10] = root_linear_velocity
    root_state[:, 10:13] = root_angular_velocity

    replay_env.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
    replay_env.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
    replay_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def main():
    """Replay the selected demo motion."""
    motion_path = _resolve_motion_path(args_cli.robot, args_cli.index)
    print(f"[INFO] Replaying motion: {motion_path}")

    env = None
    with tempfile.TemporaryDirectory(prefix=f"{args_cli.robot}_replay_") as motion_dir:
        replay_motion_path = Path(motion_dir) / motion_path.name
        shutil.copy2(motion_path, replay_motion_path)

        try:
            env_cfg = _configure_replay_env(motion_dir)
            env = gym.make(TASK_NAME, cfg=env_cfg)
            replay_env = env.unwrapped
            replay_env.motion_loader._cycle_motion = args_cli.loop

            env.reset()
            env_ids = torch.arange(replay_env.num_envs, dtype=torch.long, device=replay_env.device)
            zero_actions = torch.zeros(env.action_space.shape, dtype=torch.float32, device=replay_env.device)
            duration = max(float(replay_env.motion_loader._motion_durations[0]), replay_env.step_dt)
            motion_time = 0.0

            print(
                "[INFO] Duration: "
                f"{duration:.3f}s | dt: {replay_env.step_dt:.4f}s | loop: {args_cli.loop}"
            )

            while simulation_app.is_running():
                frame_start = time.time()
                if not args_cli.loop and motion_time > duration:
                    break

                with torch.inference_mode():
                    _write_motion_state(replay_env, motion_time, env_ids)
                    if hasattr(replay_env.sim, "render"):
                        replay_env.sim.render()
                    env.step(zero_actions)

                motion_time += replay_env.step_dt
                if args_cli.loop:
                    motion_time = motion_time % duration

                sleep_time = replay_env.step_dt - (time.time() - frame_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
