"""Probe the DiSECt bridge directly: descend the knife and print force vs height.
Run in any python: python probe_bridge.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from disect_bridge.client import DisectClient

c = DisectClient()
c.reset()
y = 0.078
v = -0.05
step_dy = -0.0015  # descend 1.5 mm per probe step
for i in range(50):
    out = c.step([0.0, y, 0.0], [0.0, v, 0.0], substeps=100)
    if i % 2 == 0:
        print(f"y={y:.4f} force_norm={out['force_norm']:8.3f} "
              f"completion={out['cut_completion']:.3f} t={out['sim_time']:.3f}")
    y += step_dy
print("PROBE_DONE")
c.close()
