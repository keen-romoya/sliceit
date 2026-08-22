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
    for step in range(args.steps):
        with torch.inference_mode():
            sampled, outputs = runner.agent.act(obs, None, timestep=0, timesteps=args.steps)
            actions = outputs.get("mean_actions", sampled)
        obs, rew, terminated, truncated, info = wrapped.step(actions)
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
        if step % 40 == 0:
            blade_h, _ = env._blade_state()
            print(f"[play {step:4d}] force max {env._latest_force.max():6.2f} N | "
                  f"completion {env._cut_completion.mean():.2f} | reward {rew.mean():.3f} | "
                  f"act0 {actions[:, 0].mean():.2f} act1 {actions[:, 1].mean():.2f} "
                  f"blade_h {blade_h[0]:.3f}")
    writer.close()
    print(f"wrote {args.out}")
    print("PLAY_OK")
    wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
