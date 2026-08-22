"""DiSECt co-simulation server.

Replaces SliceIt's ROS service interface (/disect/load, /disect/reset,
/disect/step_simulation + knife odometry topic) with a TCP JSON-lines
protocol, so the high-fidelity cutting simulator can be stepped from an
Isaac Lab environment (or anything else) without ROS.

Run inside the DiSECt conda env (NOT Isaac's python):

    conda activate disect
    python server.py --config examples/config/ansys_sphere_apple.json \
        --disect-root /root/DiSECt-sliceit [--params best_optuna.pkl]

Protocol (one JSON object per line, response per request):
    {"cmd": "reset"}
        -> {"ok": true}
    {"cmd": "step", "pos": [x,y,z], "vel": [vx,vy,vz], "substeps": 50}
        -> {"ok": true, "force": [fx,fy,fz], "force_norm": f,
            "cut_completion": c, "sim_time": t}
    {"cmd": "info"}
        -> {"ok": true, "dt": ..., "duration": ...}

Knife pose convention matches disect_slicing.py in the Gazebo version:
the client streams the knife pose already expressed in DiSECt's cutting
frame; this server just applies it kinematically before stepping.
"""
import argparse
import json
import os
import pickle
import socketserver
import sys

import numpy as np
import torch


class DisectSession:
    def __init__(self, disect_root, config_path, params_path=None, device="cuda"):
        sys.path.insert(0, disect_root)
        os.chdir(disect_root)
        from disect.cutting import load_settings, create_sim

        self.device = device
        settings = load_settings(config_path)
        settings.sim_dt = 4e-5
        # groundtruth is only needed for calibration; reuse the ANSYS apple
        # profile so create_sim() is happy even when only co-simulating.
        if not settings.get("groundtruth", None):
            settings.groundtruth = "dataset/forces/sphere_fine_resultant_force_xyz.csv"
        best_params = None
        if params_path:
            best_params = pickle.load(open(params_path, "rb"))
            print(f"loaded calibrated params: {best_params}")
        self.settings = settings
        self.create_sim = create_sim
        self.best_params = best_params
        self.sim = None
        self.reset()

    def reset(self):
        sim, _params = self.create_sim(
            self.settings, "disect_bridge", requires_grad=False,
            best_params=self.best_params, device=self.device,
            verbose=False, shared_params=True)
        sim.init_parameters()
        sim.init_sim_structures_()
        del sim.model
        sim.create_model_()
        sim.state = sim.model.state()
        sim.assign_parameters()
        self.sim = sim
        self.initial_knife_pos = None

    def step(self, pos, vel, substeps):
        """Kinematically drive the knife, advance the FEM sim, return force."""
        from disect.cutting import ConstantLinearVelocityMotion

        sim = self.sim
        if self.initial_knife_pos is None:
            self.initial_knife_pos = np.asarray(pos, dtype=np.float32)
        # ConstantLinearVelocityMotion evaluates initial_pos + t * v with
        # ABSOLUTE sim time, so compensate to make the knife sit at `pos` now.
        pos_t = torch.tensor(pos, device=self.device, dtype=torch.float32)
        vel_t = torch.tensor(vel, device=self.device, dtype=torch.float32)
        sim.motion = ConstantLinearVelocityMotion(
            initial_pos=pos_t - sim.sim_time * vel_t,
            linear_velocity=vel_t)
        for _ in range(substeps):
            sim.simulation_step()
        knife_f = sim.state.knife_f
        total = torch.sum(knife_f, dim=0)
        force = total.detach().cpu().numpy().tolist()
        force_norm = float(torch.sum(torch.norm(knife_f, dim=1)).item())
        ke = sim.state.cut_spring_ke
        cut_completion = float(1.0 - (torch.mean(ke) / sim.model.cut_spring_stiffness.mean()).item()) if ke.numel() else 1.0
        return {
            "force": force,
            "force_norm": force_norm,
            "cut_completion": max(0.0, min(1.0, cut_completion)),
            "sim_time": sim.sim_time,
        }

    def mesh_points(self):
        """Current deformed render vertices (DiSECt frame, y-up): FEM nodes
        plus the interpolated cut-edge points that carry the re-tessellated
        band where the surface crosses the cut plane (same construction as
        DiSECt's own visualizer)."""
        sim = self.sim
        ps = sim.state.particle_q.detach().cpu().numpy()
        ids = sim.model.cut_edge_indices.detach().cpu().numpy()
        coords = sim.model.cut_edge_coords.detach().cpu().numpy()
        cut_v = (ps[ids[:, 0]] * (1.0 - coords)[:, None]
                 + ps[ids[:, 1]] * coords[:, None])
        return np.vstack([ps, cut_v]).round(5).tolist()

    def mesh_topology(self):
        """Watertight render topology: base surface + cut-band triangles +
        cut-face caps (above/below), indexed over [nodes; cut-edge points]."""
        sim = self.sim
        n = sim.state.particle_q.shape[0]
        base = np.asarray(sim.builder.tri_indices).reshape(-1, 3)
        band = sim.model.cut_tri_indices.detach().cpu().numpy().reshape(-1, 3)
        above = sim.model.cut_virtual_tri_indices_above_cut.detach().cpu().numpy().reshape(-1, 3) + n
        below = sim.model.cut_virtual_tri_indices_below_cut.detach().cpu().numpy().reshape(-1, 3) + n
        tris = np.vstack([base, band, above, below])
        return {"tris": tris.tolist(), "points": self.mesh_points()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="examples/config/ansys_sphere_apple.json")
    ap.add_argument("--disect-root", default="/root/DiSECt-sliceit")
    ap.add_argument("--params", default=None, help="pickled Optuna best_params")
    ap.add_argument("--port", type=int, default=8299)
    args = ap.parse_args()

    session = DisectSession(args.disect_root, args.config, args.params)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                try:
                    req = json.loads(line)
                    cmd = req.get("cmd")
                    if cmd == "reset":
                        session.reset()
                        resp = {"ok": True}
                    elif cmd == "step":
                        out = session.step(req["pos"], req["vel"],
                                           int(req.get("substeps", 50)))
                        if req.get("include_mesh"):
                            out["mesh_points"] = session.mesh_points()
                        resp = {"ok": True, **out}
                    elif cmd == "mesh_topology":
                        resp = {"ok": True, **session.mesh_topology()}
                    elif cmd == "info":
                        resp = {"ok": True, "dt": session.settings.sim_dt,
                                "duration": session.settings.sim_duration}
                    else:
                        resp = {"ok": False, "error": f"unknown cmd {cmd}"}
                except Exception as e:  # report, keep serving
                    resp = {"ok": False, "error": repr(e)}
                self.wfile.write((json.dumps(resp) + "\n").encode())
                self.wfile.flush()

    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as srv:
        srv.allow_reuse_address = True
        print(f"disect bridge serving on 127.0.0.1:{args.port}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
