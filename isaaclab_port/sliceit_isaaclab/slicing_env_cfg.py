"""Configuration for the SliceIt slicing task ported from Gazebo to Isaac Lab."""
from isaaclab_assets.robots.universal_robots import UR10e_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class SlicingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4                # policy at 30 Hz over a 120 Hz physics step
    episode_length_s = 8.0
    action_space = 2              # [v_down, v_slice] task-space velocity commands
    observation_space = 13        # depth_err(1) ee_vel(3) completion(1) last_action(2) force_hist(6)
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=3.0, replicate_physics=True)

    robot: ArticulationCfg = UR10e_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # cutting scene geometry (env-local frame)
    board_pos = (0.7, 0.0, 0.01)
    food_pos = (0.7, 0.0, 0.045)      # cucumber-proxy center
    food_size = (0.15, 0.04, 0.05)    # x-long cylinder proxy as a box
    food_surface_height = 0.07        # top of food, where cutting force engages
    completion_depth = 0.045          # blade travel through material = cut done

    # knife: thin blade parented to the wrist; edge offset in EE frame
    knife_size = (0.20, 0.008, 0.11)
    blade_edge_offset = (0.0, 0.0, 0.26)  # from wrist_3_link origin to blade edge, along tool z

    # control (mirrors SliceIt's compliant task-space commands)
    max_down_velocity = 0.05      # m/s, matches their pressing_velocity range
    max_slice_velocity = 0.10     # m/s along the blade axis
    ik_damping = 0.05

    # force model: "profile" (vectorized, distilled from DiSECt) or
    # "bridge" (live DiSECt co-sim over TCP; requires num_envs == 1)
    force_model = "profile"
    force_profile_path = "force_profile_apple.npz"
    bridge_host = "127.0.0.1"
    bridge_port = 8299
    bridge_substeps = 100         # DiSECt steps (dt 4e-5) per policy step
    # maps Isaac blade-edge height to DiSECt knife position:
    # surface offset (isaac 0.07 vs disect sphere top 0.05) + knife spine
    # half-height (DiSECt knife pos is the spine center, edge = pos + 0.029)
    bridge_height_offset = 0.049

    # reward weights — port of cost_utils.slicing_with_vel
    w_dist = 1.0
    w_force = 1.0
    w_jerk = 0.2
    w_vel = 0.2
    cost_done = 10.0
    cost_collision = -10.0
    cost_step = -0.01
    max_force = 50.0              # |F| beyond this terminates as collision (N)
    force_hist_len = 6
