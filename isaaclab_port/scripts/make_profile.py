"""Distill a DiSECt force recording into the depth-indexed profile used by
ProfileForceModel. Reads one of the dataset force CSVs (constant downward
knife velocity) and writes force_profile_apple.npz with depth[m], force[N].

Run from the DiSECt repo root (any python with numpy/pandas):
    python make_profile.py --csv dataset/forces/sphere_fine_resultant_force_xyz.csv \
        --velocity 0.05 --out force_profile_apple.npz
"""
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="dataset/forces/sphere_fine_resultant_force_xyz.csv")
parser.add_argument("--velocity", type=float, default=0.05, help="knife speed [m/s]")
parser.add_argument("--contact-threshold", type=float, default=0.5, help="N")
parser.add_argument("--out", default="force_profile_apple.npz")
args = parser.parse_args()

# LS-DYNA rcforc export: title row, then Time,X-force,Time,Y-force,Time,Z-force
df = pd.read_csv(args.csv, skiprows=1)
vals = df.to_numpy(dtype=float)
time = vals[:, 0]
force = np.linalg.norm(vals[:, [1, 3, 5]], axis=1)

contact = np.argmax(force > args.contact_threshold)
time = time[contact:] - time[contact]
force = force[contact:]
depth = time * args.velocity

np.savez(args.out, depth=depth.astype(np.float32), force=force.astype(np.float32))
print(f"wrote {args.out}: {len(depth)} samples, depth 0..{depth[-1]:.3f} m, "
      f"peak force {force.max():.1f} N")
