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

        food = sim_utils.CuboidCfg(
            size=self.cfg.food_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.55, 0.20)))
        food.func("/World/envs/env_0/Food", food, translation=self.cfg.food_pos)

        knife = sim_utils.CuboidCfg(
            size=self.cfg.knife_size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.78)))
        knife.func("/World/envs/env_0/Robot/wrist_3_link/Knife", knife,
                   translation=(0.0, 0.0, 0.16))

        light = sim_utils.DomeLightCfg(intensity=2200.0)
        light.func("/World/light", light)

        self.scene.clone_environments(copy_from_source=False)

    # -------------------------------------------------------------- act/step

    def _pre_physics_step(self, actions: torch.Tensor):
        self._last_action = actions.clamp(-1.0, 1.0)
        # task-space velocity command: press down (world -z), slice along blade x
        v = torch.zeros((self.num_envs, 6), device=self.device)
        v[:, 2] = -self.cfg.max_down_velocity * (0.5 * (self._last_action[:, 0] + 1.0))
        v[:, 0] = self.cfg.max_slice_velocity * self._last_action[:, 1]
        self._cmd_twist = v

    def _apply_action(self):
        # damped least-squares IK: qdot = J^T (J J^T + lambda I)^-1 v
        jac = self._robot.root_physx_view.get_jacobians()[:, self._ee_body_idx - 1, :, :]
        jac = jac[:, :, :6]
        jjt = jac @ jac.transpose(1, 2)
        lam = (self.cfg.ik_damping ** 2) * torch.eye(6, device=self.device).unsqueeze(0)
        qdot = jac.transpose(1, 2) @ torch.linalg.solve(jjt + lam, self._cmd_twist.unsqueeze(-1))
        q_target = self._robot.data.joint_pos[:, :6] + qdot.squeeze(-1) * self.cfg.sim.dt
        self._robot.set_joint_position_target(q_target, joint_ids=self._arm_joint_ids)

        # cutting reaction force on the wrist from the DiSECt-derived model
        blade_h, blade_vel = self._blade_state()
        if self._force_model is not None:
            f_up, self._cut_completion = self._force_model.step(blade_h, blade_vel[:, 2])
        else:
            f_up, self._cut_completion = self._bridge_force(blade_h, blade_vel)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = f_up  # material pushes back up on the blade
        torques = torch.zeros_like(forces)
        self._robot.set_external_force_and_torque(
            forces, torques, body_ids=[self._ee_body_idx])
        self._latest_force = f_up

    def _bridge_force(self, blade_h, blade_vel):
        ee_pos = self._ee_pos_env()[0]
        # DiSECt cutting frame: y is up, knife descends along -y
        pos = [float(ee_pos[0] - self.cfg.food_pos[0]), float(blade_h[0]), float(ee_pos[1])]
        vel = [float(blade_vel[0, 0]), float(blade_vel[0, 2]), float(blade_vel[0, 1])]
        out = self._bridge.step(pos, vel, substeps=self.cfg.bridge_substeps)
        f = torch.tensor([out["force_norm"]], device=self.device)
        c = torch.tensor([out["cut_completion"]], device=self.device)
        return f, c

    # ------------------------------------------------------------------- obs

    def _ee_pos_env(self):
        return (self._robot.data.body_pos_w[:, self._ee_body_idx]
                - self.scene.env_origins)

    def _blade_state(self):
        ee_pos = self._ee_pos_env()
        blade_h = ee_pos[:, 2] - self._blade_offset[:, 2]
        blade_vel = self._robot.data.body_lin_vel_w[:, self._ee_body_idx]
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
        terminated = self._goal_reached | self._collision
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # start pose: blade above the food (tuned for UR10e reach)
        joint_pos[:, :6] = torch.tensor(
            [0.0, -1.2, 1.6, -1.97, -1.57, 0.0], device=self.device)
        self._robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._last_action[env_ids] = 0.0
        self._prev_ee_vel[env_ids] = 0.0
        self._prev_ee_acc[env_ids] = 0.0
        self._force_hist[env_ids] = 0.0
        self._cut_completion[env_ids] = 0.0
        self._goal_reached[env_ids] = False
        self._collision[env_ids] = False
        self._latest_force = torch.zeros(self.num_envs, device=self.device)
        if self._force_model is not None:
            self._force_model.reset(env_ids)
        elif self._bridge is not None:
            self._bridge.reset()
