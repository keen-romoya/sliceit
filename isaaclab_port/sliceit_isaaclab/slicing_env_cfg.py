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

    # cutting scene geometry (env-local frame), centered under the EE's
    # start-pose position so the blade meets the food. The cut is crosswise:
    # food long along x, blade plane at cut_plane_x, slicing motion along y.
    board_pos = (0.87, 0.174, 0.01)
    cut_plane_x = 0.868               # EE start x — where the blade descends
    food_y = 0.174
    food_size = (0.15, 0.04, 0.05)    # full x-long block before the split
    food_left_center = (0.8065, 0.174, 0.045)   # [0.745, 0.868]
    food_left_size = (0.123, 0.04, 0.05)
    # slice sits one blade-thickness past the cut plane so the blade occupies
    # the kerf it cuts and never interpenetrates solid material
    food_right_center = (0.8895, 0.174, 0.045)  # [0.876, 0.903] — the slice
    food_right_size = (0.027, 0.04, 0.05)
    slice_separation = 0.018          # how far the cut slice drifts at completion 1.0
    food_surface_height = 0.07        # top of food, where cutting force engages
    completion_depth = 0.045          # blade travel through material = cut done

    # knife: blade long along y (crosswise cut), thin along x; its top meets
    # the wrist flange (~0.07 below the wrist origin)
    knife_size = (0.008, 0.20, 0.17)
    blade_edge_offset = (0.0, 0.0, 0.20)  # wrist_3_link origin to blade edge (world down)

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
