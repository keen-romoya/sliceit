"""Vectorized phenomenological cutting-force model distilled from DiSECt.

The live bridge (server.py/client.py) is single-instance: one DiSECt FEM
simulation per process. For massively parallel RL in Isaac Lab we instead
distill a calibrated DiSECt run into a depth-indexed force profile with a
damage state per environment — the same role the "fast simulator" plays in
the SliceIt dual-simulator architecture, but derived from DiSECt output
rather than Gazebo contact springs.

Model: while the blade moves down through the material, resistance follows
the recorded force-vs-depth curve scaled by remaining "spring integrity";
integrity decays with accumulated downward travel (crack propagation), so a
second pass over cut material is nearly free — mirroring DiSECt's weakening
cutting springs.
"""
import numpy as np
import torch


class ProfileForceModel:
    def __init__(self, profile_npz, num_envs, device,
                 surface_height=0.05, completion_depth=0.045):
        """profile_npz: .npz with arrays depth[m] (from first contact) and force[N]."""
        data = np.load(profile_npz)
        self.depths = torch.tensor(data["depth"], device=device, dtype=torch.float32)
        self.forces = torch.tensor(data["force"], device=device, dtype=torch.float32)
        self.surface_height = surface_height
        self.completion_depth = completion_depth
        self.num_envs = num_envs
        self.device = device
        self.integrity = torch.ones(num_envs, device=device)
        self.max_depth_seen = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.integrity[env_ids] = 1.0
        self.max_depth_seen[env_ids] = 0.0

    def step(self, blade_height, blade_vel_y):
        """blade_height: (N,) world height of blade edge. Returns (force_y, cut_completion)."""
        depth = (self.surface_height - blade_height).clamp(min=0.0)
        new_material = (depth - self.max_depth_seen).clamp(min=0.0)
        self.max_depth_seen = torch.maximum(self.max_depth_seen, depth)

        idx = torch.searchsorted(
            self.depths, depth.clamp(max=self.depths[-1])).clamp(max=len(self.depths) - 1)
        base_force = self.forces[idx]
        # only resist while pressing into uncut material
        pressing = (blade_vel_y < 0.0).float()
        cutting = (new_material > 0.0).float()
        force_y = base_force * self.integrity * pressing * cutting
        # crack propagation: integrity decays with newly cut depth
        self.integrity = (self.integrity - new_material / max(self.completion_depth, 1e-6)).clamp(min=0.0)
        completion = (self.max_depth_seen / self.completion_depth).clamp(max=1.0)
        return force_y, completion
