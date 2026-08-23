"""Calibrate DiSECt against the REAL SliceIt force recordings (osx_dataset).

The fork's own real2sim flow: Optuna global search over cutting parameters,
then Adam gradient refinement using DiSECt's differentiability, fitting the
simulated knife-force profile to a real UR-robot recording.

Run from the DiSECt fork root in the disect conda env:
    python calibrate_real.py --veggie cucumber --trials 60 --adam-iters 15
"""
import argparse
import os
import pickle

import torch
torch.autograd.set_detect_anomaly(False, check_nan=True)

from tensorboardX import SummaryWriter
from disect.cutting import load_settings, save_settings, optuna_trainer, adam_trainer, create_sim

parser = argparse.ArgumentParser()
parser.add_argument("--veggie", default="cucumber",
                    choices=["cucumber", "potato", "tomato"])
parser.add_argument("--trials", type=int, default=60)
parser.add_argument("--adam-iters", type=int, default=15)
args = parser.parse_args()

GROUNDTRUTH = {
    "cucumber": "osx_dataset/calibrated/cucumber_3_05.npy",
    "potato": "osx_dataset/calibrated/potato_01.npy",
    "tomato": "osx_dataset/calibrated/tomato_01.npy",
}

settings = load_settings(f"examples/config/osx/ansys_{args.veggie}.json")
settings.sim_dt = 4e-5
settings.initial_y = 0.059 / 2. + settings.veggie_height + 0.001
settings.velocity_y = -0.020
# the config's groundtruth path points inside the SliceIt docker; use the
# recovered dataset relative to the repo root
settings.groundtruth = GROUNDTRUTH[args.veggie]

experiment_name = f"real_calib_{args.veggie}"
logger = SummaryWriter(logdir=f"log/{experiment_name}")
os.makedirs(f"log/{experiment_name}/plots", exist_ok=True)
os.makedirs(f"log/{experiment_name}/params", exist_ok=True)
save_settings(settings, f"log/{experiment_name}/settings.json")

sim, parameters = create_sim(settings, experiment_name, requires_grad=False,
                             device="cuda", verbose=False, shared_params=True)
best_params = optuna_trainer(sim, parameters, logger, n_trials=args.trials)
print("Optuna best:", best_params)
optuna_pkl = f"log/{experiment_name}/best_optuna_optimized_tensors.pkl"
pickle.dump(best_params, open(optuna_pkl, "wb"))
print("OPTUNA_DONE")

if args.adam_iters > 0:
    settings.sim_dt = 2e-5  # finer dt for the gradient phase (fewer NaNs)
    sim, parameters = create_sim(settings, experiment_name, requires_grad=True,
                                 best_params=best_params, device="cuda", verbose=True)
    adam_trainer(sim, logger, learning_rate=0.5, iterations=args.adam_iters,
                 previous_best=optuna_pkl)
    print("ADAM_DONE")
print("CALIB_REAL_DONE")
