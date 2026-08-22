"""Roll out a trained checkpoint and record a video of the slicing env.

    /root/IsaacLab/isaaclab.sh -p scripts/play_video.py --headless \
        --enable_cameras --checkpoint <path/to/best_agent.pt> \
        [--num_envs 4] [--steps 240] [--out rollout.mp4]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", default="rollout.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import imageio.v2 as imageio
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from sliceit_isaaclab.slicing_env import SlicingEnv
from sliceit_isaaclab.slicing_env_cfg import SlicingEnvCfg


def main():
    cfg = SlicingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # rgb_array rendering uses cfg.viewer for its camera, not the GUI viewport
    cfg.viewer.eye = (1.55, 1.05, 0.55)
    cfg.viewer.lookat = (0.87, 0.174, 0.14)
    env = SlicingEnv(cfg, render_mode="rgb_array")
    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg_path = os.path.join(os.path.dirname(__file__), "..",
                                  "sliceit_isaaclab", "agents", "skrl_ppo_cfg.yaml")
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    runner = Runner(wrapped, agent_cfg)
    runner.agent.load(args.checkpoint)

    obs, _ = wrapped.reset()
    writer = imageio.get_writer(args.out, fps=30, quality=8)

    # numeric contact telemetry: verify blade/food intersection from sim
    # state per frame instead of judging rendered pixels
    telemetry = []
    fl_c, fl_s = cfg.food_left_center, cfg.food_left_size
    food_aabb = (fl_c[0] - fl_s[0] / 2, cfg.food_right_center[0] + cfg.food_right_size[0] / 2,
                 fl_c[1] - fl_s[1] / 2, fl_c[1] + fl_s[1] / 2,
                 fl_c[2] - fl_s[2] / 2, fl_c[2] + fl_s[2] / 2)
    for step in range(args.steps):
        with torch.inference_mode():
            sampled, outputs = runner.agent.act(obs, None, timestep=0, timesteps=args.steps)
            actions = outputs.get("mean_actions", sampled)
        obs, rew, terminated, truncated, info = wrapped.step(actions)
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
        # per-frame numeric telemetry (env 0)
        blade_h, _ = env._blade_state()
        ee = env._ee_pos_env()[0]
        kx, ky, kz = float(ee[0]), float(ee[1]), float(blade_h[0])
        bx0, bx1 = kx - cfg.knife_size[0] / 2, kx + cfg.knife_size[0] / 2
        by0, by1 = ky - cfg.knife_size[1] / 2, ky + cfg.knife_size[1] / 2
        overlap_xy = (bx1 > food_aabb[0] and bx0 < food_aabb[1]
                      and by1 > food_aabb[2] and by0 < food_aabb[3])
        penetration = max(0.0, food_aabb[5] - kz) if overlap_xy else 0.0
        telemetry.append((step, kx, ky, kz, penetration,
                          float(env._cut_completion[0]), float(env._latest_force[0])))
        if step % 40 == 0:
            print(f"[play {step:4d}] force {env._latest_force[0]:6.2f} N | "
                  f"completion {env._cut_completion[0]:.2f} | reward {rew.mean():.3f} | "
                  f"blade edge=({kx:.3f},{ky:.3f},{kz:.3f}) "
                  f"penetration {penetration*1000:.0f} mm")
    writer.close()
    print(f"wrote {args.out}")

    import csv
    with open(args.out.replace(".mp4", "_telemetry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "blade_x", "blade_y", "blade_edge_z",
                    "penetration_m", "completion", "force_n"])
        w.writerows(telemetry)
    pen_frames = [t for t in telemetry if t[4] > 0]
    if pen_frames:
        deepest = max(pen_frames, key=lambda t: t[4])
        print(f"CONTACT: {len(pen_frames)}/{len(telemetry)} frames with blade-food "
              f"intersection; deepest {deepest[4]*1000:.0f} mm at frame {deepest[0]}")
    else:
        print("CONTACT: NONE — blade never intersects the food AABB")
    print("PLAY_OK")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
