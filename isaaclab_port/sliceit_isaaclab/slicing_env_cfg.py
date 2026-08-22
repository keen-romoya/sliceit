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
    max_down_velocity = 0.04      # m/s (reduced: calibrated material is stiff)
    max_slice_velocity = 0.10     # m/s along the blade axis
    ik_damping = 0.05

    # force model: "profile" (vectorized, distilled from DiSECt) or
    # "bridge" (live DiSECt co-sim over TCP; requires num_envs == 1)
    force_model = "profile"
    force_profile_path = "force_profile_apple.npz"
    bridge_host = "127.0.0.1"
    bridge_port = 8299
    bridge_substeps = 100         # DiSECt steps (dt 4e-5) per policy step
    # maps Isaac blade-edge height to DiSECt knife position: scene surface
    # offset only (isaac food top 0.07 <-> disect force onset at y=0.05,
    # verified by servo-tracked force probing; the old 0.049 was fit against
    # the pre-servo motion bug)
    bridge_height_offset = 0.02

    # reward weights — port of cost_utils.slicing_with_vel
    w_dist = 1.5
    w_force = 1.0
    w_jerk = 0.2
    w_vel = 0.2
    cost_done = 10.0
    cost_collision = -10.0
    cost_step = -0.02
    # calibrated-apple-appropriate force budget: the LS-DYNA-calibrated
    # material genuinely needs ~70-105 N to cut through (force is depth-
    # dominated); 50 N was a soft-produce number that made completion
    # impossible past ~15 mm
    max_force = 120.0
    # s-shaped force penalty centered for the calibrated material's working
    # range (~10-45 N) so the gradient rewards easing off, not just avoidance
    force_penalty_center = 40.0
    force_penalty_scale = 10.0
    force_hist_len = 6
