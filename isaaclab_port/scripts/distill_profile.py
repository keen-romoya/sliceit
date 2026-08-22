"""Distill a force-vs-depth profile from the live (calibrated) DiSECt bridge.

Runs a constant-speed reference descent through the material and records the
knife force at each depth. The result replaces the LS-DYNA-derived profile so
the vectorized training model matches what the bridge will do at rollout.

Run in the disect conda env with the bridge server up:
    python distill_profile.py --out /root/sliceit/isaaclab_port/force_profile_apple.npz
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from disect_bridge.client import DisectClient

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="force_profile_apple.npz")
parser.add_argument("--speed", type=float, default=0.05, help="reference press speed [m/s]")
parser.add_argument("--surface", type=float, default=0.05,
                    help="DiSECt height of force onset — depth 0 in the profile")
parser.add_argument("--start", type=float, default=0.06,
                    help="DiSECt height where the descent begins")
parser.add_argument("--depth", type=float, default=0.085, help="total recorded travel [m]")
args = parser.parse_args()

c = DisectClient()
c.reset()

window = 100 * 4e-5              # bridge step window [s]
dy = -args.speed * window        # real-time descent per call
y = args.start
depths, forces = [], []
n_calls = int((args.depth + (args.start - args.surface)) / abs(dy))
print(f"distilling: {n_calls} calls at {args.speed} m/s, dy={dy * 1000:.2f} mm")
for i in range(n_calls):
    out = c.step([0.0, y, 0.0], [0.0, -args.speed, 0.0], substeps=100)
    depths.append(args.surface - y)
    forces.append(out["force_norm"])
    y += dy
    if i % 40 == 0:
        print(f"  depth={depths[-1] * 1000:6.1f} mm force={forces[-1]:6.2f} N "
              f"compl={out['cut_completion']:.2f}")
c.close()

depth = np.asarray(depths, dtype=np.float32)
force = np.asarray(forces, dtype=np.float32)
keep = depth >= 0.0  # drop the approach above the surface
depth, force = depth[keep], force[keep]
# light smoothing: the FEM force has substep-scale ripple
kernel = np.ones(5) / 5.0
force = np.convolve(force, kernel, mode="same").astype(np.float32)
np.savez(args.out, depth=depth, force=force,
         reference_speed=np.float32(args.speed))
print(f"wrote {args.out}: {len(depth)} samples, peak {force.max():.1f} N "
      f"at depth {depth[force.argmax()] * 1000:.0f} mm")
