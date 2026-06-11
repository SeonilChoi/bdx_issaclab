import os

from bdx.robots import BDX_CFG, ISAACLAB_ASSETS_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


BDX_NUM_JOINTS = 16
BDX_BASE_AMP_OBSERVATION_SPACE = 1 + 6 + 3 + 3 + BDX_NUM_JOINTS * 6 + BDX_NUM_JOINTS + 3 + 3
BDX_POLICY_OBSERVATION_SPACE = 6 + 3 + BDX_NUM_JOINTS * 6 + BDX_NUM_JOINTS + 3 + 3
BDX_AMP_HEAD_NECK_JOINT_NAMES = ("head_pitch", "head_yaw")
BDX_AMP_HEAD_NECK_OBSERVATION_REPEATS = 2
BDX_AMP_HEAD_NECK_FEATURE_SPACE = len(BDX_AMP_HEAD_NECK_JOINT_NAMES) * (6 + 1)
BDX_AMP_OBSERVATION_SPACE = (
    BDX_BASE_AMP_OBSERVATION_SPACE
    + BDX_AMP_HEAD_NECK_FEATURE_SPACE * BDX_AMP_HEAD_NECK_OBSERVATION_REPEATS
)


@configclass
class BdxAmpEnvCfg(DirectRLEnvCfg):
    # environment configuration
    decimation = 4
    episode_length_s = 15.0

    # space configuration
    action_space = BDX_NUM_JOINTS  # 16D vector of joint positions
    observation_space = BDX_POLICY_OBSERVATION_SPACE + 3
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = BDX_AMP_OBSERVATION_SPACE
    base_amp_observation_space = BDX_BASE_AMP_OBSERVATION_SPACE
    amp_head_neck_joint_names = BDX_AMP_HEAD_NECK_JOINT_NAMES
    amp_head_neck_observation_repeats = BDX_AMP_HEAD_NECK_OBSERVATION_REPEATS
    action_scale = 0.6

    command_linear_x_range = (-0.2719, 0.2745)
    command_linear_y_range = (-0.2567, 0.2609)
    command_yaw_range = (-0.2828, 0.2828)
    command_linear_velocity_scale = 0.5
    command_angular_velocity_scale = 0.25
    sample_commands_from_reference = True
    use_average_velocities = True
    average_velocity_window_s = 0.6
    command_tracking_sigma = 0.05
    linear_velocity_xy_reward_scale = 0.5
    angular_velocity_z_reward_scale = 0.25
    torque_reward_scale = -0.000025
    action_rate_reward_scale = 0.0
    stand_still_reward_scale = 0.0
    stand_still_command_threshold = 0.01

    early_termination = True
    termination_height = 0.2
    reference_body = "pelvis"
    reset_strategy = "random"
    use_isaac_imu_sensor = True
    imu_backend = "isaaclab"
    imu_body = "trunk"
    imu_sensor_name = "mti3_imu"
    imu_prim_path = "/World/envs/env_.*/Robot/trunk"
    imu_parent_path_template = "/World/envs/env_{env_id}/Robot/{body}"
    imu_sensor_period_s = 0.0
    imu_linear_acceleration_filter_size = 1
    imu_angular_velocity_filter_size = 1
    imu_orientation_filter_size = 1
    imu_translation = (0.0, 0.0, 0.0)
    imu_orientation_wxyz = (1.0, 0.0, 0.0, 0.0)
    imu_read_gravity = True
    imu_latency_s = 0.0
    mti3_gyro_bias_walk_std_rad_s = 0.0
    mti3_accel_bias_walk_std_m_s2 = 0.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
        ),
    )

    # robot
    robot: ArticulationCfg = BDX_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # custom parameters
    dt = 1 / 240 * 4
    motion_folder_path = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "bdx")
