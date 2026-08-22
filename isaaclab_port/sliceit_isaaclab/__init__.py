"""SliceIt slicing task for Isaac Lab (port of the Gazebo ur3e_openai env)."""
import gymnasium as gym

from . import agents

gym.register(
    id="SliceIt-Slicing-Direct-v0",
    entry_point=f"{__name__}.slicing_env:SlicingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slicing_env_cfg:SlicingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
