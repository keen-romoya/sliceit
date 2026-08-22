"""SliceIt slicing task on Isaac Lab (Direct workflow).

Port of ur3e_openai's UR3eSlicingEnv (Gazebo + ROS) to Isaac Lab:
- UR10e arm with a blade parented to the wrist (UR3e swap is a config change
  once its USD is converted from the SliceIt URDF).
- Task-space velocity actions (press + slice), damped-least-squares IK,
  standing in for their cartesian compliance controller.
- Cutting force from DiSECt, either distilled ("profile", vectorized) or
  live over the TCP bridge ("bridge", single env) — the same dual-simulator
  split as the original SliceIt architecture.
- Reward is a port of cost_utils.slicing_with_vel: distance progress,
  s-shaped force penalty, jerk penalty, velocity shaping, terminal bonuses.
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .slicing_env_cfg import SlicingEnvCfg


class SlicingEnv(DirectRLEnv):
    cfg: SlicingEnvCfg

    def __init__(self, cfg: SlicingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self._ee_body_idx = self._robot.find_bodies("wrist_3_link")[0][0]
        self._arm_joint_ids = self._robot.find_joints(".*")[0][:6]

        n = self.num_envs
        dev = self.device
        self._last_action = torch.zeros((n, 2), device=dev)
        self._prev_ee_vel = torch.zeros((n, 3), device=dev)
        self._prev_ee_acc = torch.zeros((n, 3), device=dev)
        self._force_hist = torch.zeros((n, self.cfg.force_hist_len), device=dev)
        self._cut_completion = torch.zeros(n, device=dev)
        self._goal_reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._collision = torch.zeros(n, dtype=torch.bool, device=dev)
        self._blade_offset = torch.tensor(self.cfg.blade_edge_offset, device=dev).repeat(n, 1)

        if self.cfg.force_model == "bridge":
            assert self.num_envs == 1, "live DiSECt bridge supports a single env"
            from disect_bridge.client import DisectClient
            self._bridge = DisectClient(self.cfg.bridge_host, self.cfg.bridge_port)
            self._force_model = None
            self._build_food_mesh_prim()
        else:
            from disect_bridge.profile_model import ProfileForceModel
            self._bridge = None
            self._force_model = ProfileForceModel(
                self.cfg.force_profile_path, n, dev,
                surface_height=self.cfg.food_surface_height,
                completion_depth=self.cfg.completion_depth)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        spawn_ground = sim_utils.GroundPlaneCfg()
        spawn_ground.func("/World/ground", spawn_ground)

        board = sim_utils.CuboidCfg(
            size=(0.35, 0.25, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)))
        board.func("/World/envs/env_0/Board", board, translation=self.cfg.board_pos)

        # food visuals. Bridge mode renders DiSECt's actual deforming FEM mesh
        # (streamed per step), so the box proxies are only for profile mode.
        self._slice_prim = None
        if self.cfg.force_model != "bridge":
            food_l = sim_utils.CuboidCfg(
                size=self.cfg.food_left_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.55, 0.20)))
            food_l.func("/World/envs/env_0/FoodLeft", food_l,
                        translation=self.cfg.food_left_center)
            food_r = sim_utils.CuboidCfg(
                size=self.cfg.food_right_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.58, 0.22)))
            food_r.func("/World/envs/env_0/FoodSliceViz", food_r,
                        translation=self.cfg.food_right_center)

        # blade visual: plain (non-physics) prim whose USD xform we write each
        # step — physics-tensor pose writes don't reach the renderer, USD
        # xform writes do (same mechanism as USD-file playback)
        knife = sim_utils.CuboidCfg(
            size=self.cfg.knife_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)))
        knife.func("/World/envs/env_0/KnifeViz", knife, translation=(0.0, 0.0, 1.5))
        self._knife_prim = None  # resolved lazily once the stage is live

        light = sim_utils.DomeLightCfg(intensity=2200.0)
        light.func("/World/light", light)

        self.scene.clone_environments(copy_from_source=False)

    # -------------------------------------------------------------- act/step

    def _pre_physics_step(self, actions: torch.Tensor):
        self._last_action = actions.clamp(-1.0, 1.0)
        # task-space velocity command: press down (world -z), saw along the
        # blade's long axis (y). A soft recentering spring keeps the sawing
        # motion bounded around the cut station (the ROS version tracked a
        # reference trajectory the same way).
        ee_y = self._ee_pos_env()[:, 1]
        v = torch.zeros((self.num_envs, 6), device=self.device)
        v[:, 2] = -self.cfg.max_down_velocity * (0.5 * (self._last_action[:, 0] + 1.0))
        v[:, 1] = (self.cfg.max_slice_velocity * self._last_action[:, 1]
                   - 3.3 * (ee_y - self.cfg.food_y))
        # kinematic board stop: the board is rigid and uncut in the dynamics,
        # so the blade edge may never be commanded below its surface
        board_top = self.cfg.board_pos[2] + 0.01
        blade_h, _ = self._blade_state()
        at_board = blade_h <= board_top + 0.002
        v[:, 2] = torch.where(at_board, v[:, 2].clamp(min=0.0), v[:, 2])
        self._board_reached = at_board
        self._cmd_twist = v

    def _apply_action(self):
        # damped least-squares IK: qdot = J^T (J J^T + lambda I)^-1 v
        # fixed-base articulation: jacobian row index is body index - 1
        jac = self._robot.data.body_link_jacobian_w.torch[:, self._ee_body_idx - 1, :, :6]
        jjt = jac @ jac.transpose(1, 2)
        lam = (self.cfg.ik_damping ** 2) * torch.eye(6, device=self.device).unsqueeze(0)
        qdot = jac.transpose(1, 2) @ torch.linalg.solve(jjt + lam, self._cmd_twist.unsqueeze(-1))
        # integrate a persistent target so PD actuators track the commanded
        # velocity; clamp target-measured error to avoid windup on contact
        q_meas = self._robot.data.joint_pos.torch[:, :6]
        self._q_target = self._q_target + qdot.squeeze(-1) * self.cfg.sim.dt
        self._q_target = q_meas + (self._q_target - q_meas).clamp(-0.08, 0.08)
        self._robot.set_joint_position_target(self._q_target, joint_ids=self._arm_joint_ids)

        # cutting reaction force on the wrist from the DiSECt-derived model,
        # gated on the blade actually being over the material laterally
        blade_h, blade_vel = self._blade_state()
        ee = self._ee_pos_env()
        fl_c, fl_s = self.cfg.food_left_center, self.cfg.food_left_size
        fr_c, fr_s = self.cfg.food_right_center, self.cfg.food_right_size
        over_material = (
            (ee[:, 0] > fl_c[0] - fl_s[0] / 2 - self.cfg.knife_size[0])
            & (ee[:, 0] < fr_c[0] + fr_s[0] / 2 + self.cfg.knife_size[0])
            & ((ee[:, 1] - self.cfg.food_y).abs() < fl_s[1] / 2 + self.cfg.knife_size[1] / 2)
        ).float()
        if self._force_model is not None:
            f_up, self._cut_completion = self._force_model.step(
                blade_h, blade_vel[:, 2], engaged=over_material)
        else:
            f_up, self._cut_completion = self._bridge_force(blade_h, blade_vel)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = f_up  # material pushes back up on the blade
        torques = torch.zeros_like(forces)
        self._robot.set_external_force_and_torque(
            forces, torques, body_ids=[self._ee_body_idx])
        self._latest_force = f_up

        # keep the blade visual glued below the wrist (USD write, env 0 only —
        # a rendering aid, invisible to physics and the policy). The physics
        # blade edge is a world-frame offset, so the visual hangs world-upright
        # with its bottom edge at the same height the force model uses.
        from pxr import Gf
        if self._knife_prim is None:
            from isaaclab.sim.utils.stage import get_current_stage
            stage = get_current_stage()
            self._knife_prim = stage.GetPrimAtPath("/World/envs/env_0/KnifeViz")
            if self.cfg.force_model != "bridge":
                self._slice_prim = stage.GetPrimAtPath("/World/envs/env_0/FoodSliceViz")
        ee_pos = self._robot.data.body_pos_w.torch[0, self._ee_body_idx]
        half_blade = 0.5 * self.cfg.knife_size[2]
        # x pinned to the kerf center: the blade renders exactly in the cut it
        # makes (EE x drift is mm-scale since x is not commanded)
        p = [self.cfg.cut_plane_x + 0.5 * self.cfg.knife_size[0], float(ee_pos[1]),
             float(ee_pos[2]) - self.cfg.blade_edge_offset[2] + half_blade]
        self._knife_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*p))
        # profile mode only: the cut slice drifts away as the cut progresses
        # (bridge mode renders the real DiSECt mesh instead)
        if self._slice_prim is not None:
            sep = self.cfg.slice_separation * float(self._cut_completion[0])
            sc = self.cfg.food_right_center
            self._slice_prim.GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(sc[0] + sep, sc[1], sc[2]))

    # ------------------------------------------------------ disect frame map
    # DiSECt: y up, knife descends along -y, blade length along z.
    # Isaac:  z up, blade length along y. Right-handed map:
    #   isaac = (d_x + cut_plane_x, -d_z + food_y, d_y + mesh_z_offset)
    _mesh_z_offset = 0.02  # DiSECt material top (0.05) -> Isaac surface (0.07)

    def _build_food_mesh_prim(self):
        import numpy as np
        from pxr import Gf, UsdGeom, Vt
        from isaaclab.sim.utils.stage import get_current_stage
        topo = self._bridge.mesh_topology()
        self._mesh_tris = np.asarray(topo["tris"], dtype=np.int64)
        stage = get_current_stage()
        mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/FoodMesh")
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray(self._mesh_tris.ravel().tolist()))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(self._mesh_tris)))
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.30, 0.55, 0.20)]))
        mesh.CreateDoubleSidedAttr(True)
        # authored (welded) normals; subdivision would re-crease the seam
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        self._food_mesh = mesh
        # weld groups: DiSECt duplicates vertices along the pre-split cut
        # surface (virtual nodes). Coincident vertices share one normal until
        # they physically separate, so the intact object renders seamless.
        pts0 = np.asarray(topo["points"], dtype=np.float64)
        keys = {}
        group_id = np.zeros(len(pts0), dtype=np.int64)
        for i, p in enumerate(np.round(pts0, 4)):
            k = (p[0], p[1], p[2])
            group_id[i] = keys.setdefault(k, len(keys))
        self._weld_group = group_id
        self._n_groups = len(keys)
        self._update_food_mesh(topo["points"])

    def _update_food_mesh(self, d_points):
        import numpy as np
        from pxr import Vt
        cx, fy, dz = self.cfg.cut_plane_x, self.cfg.food_y, self._mesh_z_offset
        d = np.asarray(d_points, dtype=np.float64)
        pts = np.stack([d[:, 0] + cx, -d[:, 2] + fy, d[:, 1] + dz], axis=1)
        self._food_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))

        # welded smooth normals: accumulate face normals per vertex, then sum
        # over weld groups whose members are still co-located (< 1.5 mm)
        tris = self._mesh_tris
        fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]],
                      pts[tris[:, 2]] - pts[tris[:, 0]])
        vn = np.zeros_like(pts)
        for c in range(3):
            np.add.at(vn, tris[:, c], fn)
        g = self._weld_group
        gsum = np.zeros((self._n_groups, 3))
        gmean = np.zeros((self._n_groups, 3))
        gcnt = np.zeros(self._n_groups)
        np.add.at(gsum, g, vn)
        np.add.at(gmean, g, pts)
        np.add.at(gcnt, g, 1.0)
        gmean /= np.maximum(gcnt, 1.0)[:, None]
        spread = np.linalg.norm(pts - gmean[g], axis=1)
        gspread = np.zeros(self._n_groups)
        np.maximum.at(gspread, g, spread)
        welded = gspread[g] < 0.0015
        vn = np.where(welded[:, None], gsum[g], vn)
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        vn = vn / np.maximum(norms, 1e-9)
        self._food_mesh.GetNormalsAttr().Set(
            Vt.Vec3fArray.FromNumpy(vn.astype(np.float32)))

        # stats for scene checks and telemetry
        self._mesh_z_top = float(pts[:, 2].max())
        self._mesh_z_min = float(pts[:, 2].min())
        self._mesh_xy_center = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        self._weld_spread = float(2.0 * gspread.max())

    def _bridge_force(self, blade_h, blade_vel):
        # inverse of the frame map above, for the knife pose streamed in
        ee = self._ee_pos_env()[0]
        disect_y = float(blade_h[0]) - self.cfg.bridge_height_offset
        pos = [0.0, disect_y, -(float(ee[1]) - self.cfg.food_y)]
        vel = [0.0, float(blade_vel[0, 2]), -float(blade_vel[0, 1])]
        out = self._bridge.step(pos, vel, substeps=self.cfg.bridge_substeps,
                                include_mesh=True)
        self._update_food_mesh(out["mesh_points"])
        ka = out.get("knife_actual")
        self._bridge_tracking_err = (
            abs(ka[1] - pos[1]) + abs(ka[2] - pos[2]) if ka else 0.0)
        f = torch.tensor([out["force_norm"]], device=self.device)
        c = torch.tensor([out["cut_completion"]], device=self.device)
        return f, c

    # ------------------------------------------------------------------- obs

    def _ee_pos_env(self):
        return (self._robot.data.body_pos_w.torch[:, self._ee_body_idx]
                - self.scene.env_origins)

    def _blade_state(self):
        ee_pos = self._ee_pos_env()
        blade_h = ee_pos[:, 2] - self._blade_offset[:, 2]
        blade_vel = self._robot.data.body_lin_vel_w.torch[:, self._ee_body_idx]
        return blade_h, blade_vel

    def _get_observations(self) -> dict:
        blade_h, blade_vel = self._blade_state()
        depth_err = ((blade_h - (self.cfg.food_surface_height - self.cfg.completion_depth))
                     / self.cfg.completion_depth).unsqueeze(-1)
        self._force_hist = torch.roll(self._force_hist, shifts=1, dims=1)
        self._force_hist[:, 0] = self._latest_force / self.cfg.max_force
        obs = torch.cat([
            depth_err,
            blade_vel / self.cfg.max_slice_velocity,
            self._cut_completion.unsqueeze(-1),
            self._last_action,
            self._force_hist,
        ], dim=-1)
        return {"policy": obs}

    # ---------------------------------------------------------------- reward

    def _get_rewards(self) -> torch.Tensor:
        blade_h, blade_vel = self._blade_state()
        dist = ((blade_h - (self.cfg.food_surface_height - self.cfg.completion_depth))
                / self.cfg.completion_depth).clamp(min=0.0)

        ee_vel = blade_vel
        acc = (ee_vel - self._prev_ee_vel) / self.dt
        jerk = torch.norm(acc - self._prev_ee_acc, dim=-1) / 500.0
        self._prev_ee_acc = acc
        self._prev_ee_vel = ee_vel

        force = self._latest_force
        r_distance = -torch.tanh(5.0 * dist)
        r_force = -1.0 / (1.0 + torch.exp(-force / 2.0 + 3.0))
        r_jerk = (-jerk).clamp(-1.0, 0.0)
        r_vel = torch.norm(ee_vel, dim=-1).clamp(0.0, 1.0) - 1.0

        w = torch.tensor([self.cfg.w_dist, self.cfg.w_force,
                          self.cfg.w_jerk, self.cfg.w_vel], device=self.device)
        w = w / w.sum()
        reward = (w[0] * r_distance + w[1] * r_force + w[2] * r_jerk + w[3] * r_vel
                  + self.cfg.cost_step)

        self._goal_reached = self._cut_completion >= 1.0
        self._collision = force > self.cfg.max_force
        reward = torch.where(self._goal_reached, reward + self.cfg.cost_done, reward)
        reward = torch.where(self._collision, reward + self.cfg.cost_collision, reward)
        return reward

    # ----------------------------------------------------------------- dones

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # the blade reaching the board ends the cut whether or not the
        # completion signal fired — there is nothing further to simulate
        terminated = self._goal_reached | self._collision | self._board_reached
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos.torch[env_ids].clone()
        # start pose: blade above the food (tuned for UR10e reach)
        joint_pos[:, :6] = torch.tensor(
            [0.0, -1.2, 1.6, -1.97, -1.57, 0.0], device=self.device)
        self._robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        if not hasattr(self, "_q_target"):
            self._q_target = torch.zeros((self.num_envs, 6), device=self.device)
        self._q_target[env_ids] = joint_pos[:, :6]
        self._last_action[env_ids] = 0.0
        self._prev_ee_vel[env_ids] = 0.0
        self._prev_ee_acc[env_ids] = 0.0
        self._force_hist[env_ids] = 0.0
        self._cut_completion[env_ids] = 0.0
        self._goal_reached[env_ids] = False
        self._collision[env_ids] = False
        if not hasattr(self, "_board_reached"):
            self._board_reached = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
        self._board_reached[env_ids] = False
        self._latest_force = torch.zeros(self.num_envs, device=self.device)
        if self._force_model is not None:
            self._force_model.reset(env_ids)
        elif self._bridge is not None:
            self._bridge.reset()
