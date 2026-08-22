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
parser.add_argument("--force-model", default="profile", choices=["profile", "bridge"])
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
    cfg.force_model = args.force_model
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
    from sliceit_isaaclab.scene_checks import SceneChecker
    checker = SceneChecker(cfg)
    contact_started = False
    telemetry = []
    if cfg.force_model == "bridge":
        checker.begin_frame(-1)
        pts_z = [p[2] for p in env._food_mesh.GetPointsAttr().Get()]
        holes = checker.watertight(env._mesh_tris.tolist(), pts_z)
        print(f"[audit] mesh watertight above bottom: "
              f"{'yes' if holes == 0 else f'NO ({holes} boundary edges)'}")
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

        # clip check: does the RENDERED blade box intersect SOLID material
        # (left block, or the slice at its current separated position)?
        eps = 1e-5
        vx0 = cfg.cut_plane_x
        vx1 = cfg.cut_plane_x + cfg.knife_size[0]
        vz0, vz1 = kz, kz + cfg.knife_size[2]
        sep = cfg.slice_separation * float(env._cut_completion[0])
        solids = [
            (fl_c[0] - fl_s[0] / 2, fl_c[0] + fl_s[0] / 2),
            (cfg.food_right_center[0] - cfg.food_right_size[0] / 2 + sep,
             cfg.food_right_center[0] + cfg.food_right_size[0] / 2 + sep),
        ]
        in_food_yz = (by1 > food_aabb[2] + eps and by0 < food_aabb[3] - eps
                      and vz0 < food_aabb[5] - eps and vz1 > food_aabb[4] + eps)
        clips = any(vx1 > s0 + eps and vx0 < s1 - eps for s0, s1 in solids) and in_food_yz
        if cfg.force_model == "bridge":
            # food geometry is the streamed DiSECt mesh: measure against it
            penetration = max(0.0, env._mesh_z_top - kz)
            clips = False  # the mesh is genuinely cut; no proxy to clip

        # scene-vs-dynamics assertions (see scene_checks.py)
        checker.begin_frame(step)
        checker.blade_vs_board((vx0, vx1, by0, by1, vz0, vz1))
        checker.blade_attachment(vz1, float(ee[2]))
        if cfg.force_model == "bridge":
            contact_started = contact_started or float(env._latest_force[0]) > 0.1
            checker.mesh_on_board(env._mesh_z_min, env._mesh_xy_center)
            checker.seam_integrity(env._weld_spread, contact_started)
        else:
            checker.solid_clip(clips)
        telemetry.append((step, kx, ky, kz, penetration,
                          float(env._cut_completion[0]), float(env._latest_force[0]),
                          int(clips)))
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
                    "penetration_m", "completion", "force_n", "clips_solid"])
        w.writerows(telemetry)
    pen_frames = [t for t in telemetry if t[4] > 0]
    clip_frames = [t for t in telemetry if t[7]]
    if pen_frames:
        deepest = max(pen_frames, key=lambda t: t[4])
        print(f"CONTACT: {len(pen_frames)}/{len(telemetry)} frames with blade-food "
              f"intersection; deepest {deepest[4]*1000:.0f} mm at frame {deepest[0]}")
    else:
        print("CONTACT: NONE — blade never intersects the food AABB")
    print(f"CLIP: {len(clip_frames)}/{len(telemetry)} frames where the rendered "
          f"blade interpenetrates solid material"
          + (f" (first at frame {clip_frames[0][0]})" if clip_frames else ""))
    print(checker.summary())
    with open(args.out.replace(".mp4", "_violations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "check", "magnitude"])
        w.writerows(checker.rows())
    print("PLAY_OK")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
