"""Train PPO on the SliceIt slicing task with skrl.

    /root/IsaacLab/isaaclab.sh -p scripts/train_ppo.py --headless \
        --num_envs 256 [--timesteps 5000]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=5000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from sliceit_isaaclab.slicing_env import SlicingEnv
from sliceit_isaaclab.slicing_env_cfg import SlicingEnvCfg


def main():
    cfg = SlicingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = SlicingEnv(cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    agent_cfg_path = os.path.join(os.path.dirname(__file__), "..",
                                  "sliceit_isaaclab", "agents", "skrl_ppo_cfg.yaml")
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["trainer"]["timesteps"] = args.timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    runner = Runner(env, agent_cfg)
    runner.run()
    print("TRAIN_OK")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
