"""Smoke test: run the SliceIt Isaac Lab env with a scripted pressing policy.

    /root/IsaacLab/isaaclab.sh -p scripts/random_policy.py --headless \
        [--num_envs 16] [--steps 200] [--force-model profile|bridge]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--force-model", default="profile", choices=["profile", "bridge"])
parser.add_argument("--profile", default="force_profile_apple.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sliceit_isaaclab.slicing_env import SlicingEnv
from sliceit_isaaclab.slicing_env_cfg import SlicingEnvCfg


def main():
    cfg = SlicingEnvCfg()
    cfg.scene.num_envs = 1 if args.force_model == "bridge" else args.num_envs
    cfg.force_model = args.force_model
    cfg.force_profile_path = args.profile

    env = SlicingEnv(cfg)
    obs, _ = env.reset()
    print(f"[smoke] envs={env.num_envs} obs={obs['policy'].shape}")

    total_reward = torch.zeros(env.num_envs, device=env.device)
    for step in range(args.steps):
        # scripted press-and-slice, mildly noisy — exercises the force model
        act = torch.zeros((env.num_envs, 2), device=env.device)
        act[:, 0] = 0.8 + 0.2 * torch.rand(env.num_envs, device=env.device)
        act[:, 1] = 0.5 * torch.sin(torch.tensor(step / 10.0))
        obs, rew, terminated, truncated, info = env.step(act)
        total_reward += rew
        if step % 20 == 0:
            f = env._latest_force
            blade_h, blade_vel = env._blade_state()
            ee = env._ee_pos_env()[0]
            print(f"[step {step:4d}] force max {f.max():6.2f} N | "
                  f"completion mean {env._cut_completion.mean():.2f} | "
                  f"reward mean {rew.mean():.3f} | "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                  f"blade_h={blade_h[0]:.3f} vz={blade_vel[0,2]:.3f}")
        if terminated.any() or truncated.any():
            done = (terminated | truncated).sum().item()
            print(f"[step {step:4d}] {done} env(s) done "
                  f"(goal={env._goal_reached.sum().item()}, "
                  f"collision={env._collision.sum().item()})")
    print(f"[smoke] done. mean episode reward so far: {total_reward.mean():.2f}")
    print("SMOKE_OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
