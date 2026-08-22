# SliceIt! → Isaac Lab port

Port of the SliceIt! robot slicing environment from Gazebo/ROS to
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (Direct workflow), keeping
the paper's dual-simulator architecture:

| SliceIt! (Gazebo/ROS) | This port (Isaac Lab) |
| --- | --- |
| UR3e + knife in Gazebo | UR10e + blade prim (UR3e USD conversion is a config swap) |
| Cartesian compliance controller (`ur_control`) | Task-space velocity actions → damped-least-squares IK |
| `/disect/{load,reset,step_simulation}` ROS services + knife odom topic | `disect_bridge/` TCP JSON-lines server/client, ROS-free |
| Gazebo contact springs as the fast sim | `ProfileForceModel`: DiSECt force-vs-depth profile + damage state, vectorized over N envs |
| `cost_utils.slicing_with_vel` reward | Same terms: tanh distance, s-shaped force penalty, jerk, velocity shaping |
| tf2rl SAC, single env | skrl PPO over parallel envs (SAC works too via skrl) |

## Layout

- `disect_bridge/server.py` — runs **in the DiSECt conda env**, wraps the
  omron-sinicx fork's `create_sim` (from `keen-romoya/DiSECt@osx_devel-modernize`)
  behind a TCP protocol. One FEM sim per process.
- `disect_bridge/client.py` / `profile_model.py` — the two force backends.
- `sliceit_isaaclab/` — `SlicingEnv` (DirectRLEnv), config, skrl agent yaml;
  registers `SliceIt-Slicing-Direct-v0`.
- `scripts/make_profile.py` — distill a DiSECt/LS-DYNA force recording into
  the `.npz` profile used by the vectorized model.
- `scripts/random_policy.py` — headless smoke test.

## Run (on the isaac-cloud instance)

```bash
# one-time: build the force profile
cd /root/DiSECt && python /root/sliceit/isaaclab_port/scripts/make_profile.py

# smoke test, 16 parallel envs with the distilled force model
cd /root/sliceit/isaaclab_port
/root/IsaacLab/isaaclab.sh -p scripts/random_policy.py --headless --num_envs 16

# live DiSECt co-simulation (terminal 1, disect conda env):
python disect_bridge/server.py --disect-root /root/DiSECt-sliceit
# terminal 2:
/root/IsaacLab/isaaclab.sh -p scripts/random_policy.py --headless --force-model bridge

# PPO training with skrl
/root/IsaacLab/isaaclab.sh -p /root/IsaacLab/scripts/reinforcement_learning/skrl/train.py \
    --task SliceIt-Slicing-Direct-v0 --num_envs 512 --headless
```

## Faithfulness notes / open items

- **Arm**: UR10e (bundled USD) instead of UR3e; convert
  `underlay_ws/src/.../ur3e` URDF with Isaac Sim's URDF importer to match the
  paper hardware.
- **Compliance**: SliceIt used a parallel position–force controller with
  learned stiffness as part of the action. Here actions are task-space
  velocities; adding stiffness dimensions to the action and an impedance law
  in `_apply_action` is the natural next step.
- **Reward weights** (`w_dist` …) mirror the code's structure; SliceIt's
  trained values live in their `ur3e_rl` ROS param yamls — tune to taste.
- **Bridge scale**: the live bridge is 1 env (one FEM sim per process); the
  distilled profile model is the massively-parallel path, exactly like the
  paper's fast-sim/high-fid split. Calibrate the profile per food with the
  fork's Optuna pipeline, re-export, retrain.
