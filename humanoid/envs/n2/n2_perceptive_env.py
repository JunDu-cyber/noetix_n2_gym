import torch
from isaacgym.torch_utils import quat_apply
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi
from humanoid.envs.n2.n2_10dof_env import N2_10dof_Env

class N2PerceptiveEnv(N2_10dof_Env):
    def _resample_commands(self, env_ids):
        """After the base class draws a new local (base-frame) velocity
        command, freeze its WORLD-frame direction/speed at this instant.
        Used by _reward_world_progress/_reward_world_heading so that turning
        away from the commanded direction (e.g. to detour around/retreat from
        an obstacle) stops being reward-neutral -- unlike heading_command,
        this target is only used by the reward, never fed back into the
        observed command, so it carries no sim2sim deployment burden."""
        super()._resample_commands(env_ids)
        if not hasattr(self, 'commands_world_dir'):
            self._init_world_progress_buffers()
        if len(env_ids) == 0:
            return
        local_vel = torch.zeros(len(env_ids), 3, device=self.device)
        local_vel[:, :2] = self.commands[env_ids, :2]
        world_vel = quat_apply_yaw(self.base_quat[env_ids], local_vel)[:, :2]
        speed = torch.norm(world_vel, dim=1)
        self.commands_world_speed[env_ids] = speed
        self.commands_world_dir[env_ids] = world_vel / speed.clamp(min=1e-6).unsqueeze(1)

    def _init_world_progress_buffers(self):
        self.commands_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self.commands_world_speed = torch.zeros(self.num_envs, device=self.device)

    def compute_observations(self):

        # ---- 单帧本体感知 (39) ----
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,        # 缩放后的命令
            self.base_ang_vel * self.obs_scales.ang_vel,       # 基座角速度
            self.projected_gravity,                            # 投影重力
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 关节位置偏差
            self.dof_vel * self.obs_scales.dof_vel,            # 关节速度
            self.actions,                                      # 当前动作
        ), dim=-1)

        # ---- 地形高度图    num_single_obs = 39 + 96 = 135 ----
        # 参考基准用 base_height_target(0.698m) 而不是硬编码 0.5:平地正常站立时
        # root_z≈base_height_target,这样 heights 才能在 0 附近居中,而不是带一个
        # ~+1.0(经 height_measurements=5 放大后)的固定偏置。
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.) * self.obs_scales.height_measurements
        obs_buf = torch.cat((obs_buf, heights), dim=-1)        # (N, num_single_obs)

        # ---- 特权观测 (critic) ----
        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.payload * 0.5,
            self.friction_coeffs,
            self.restitution_coeffs,
            self.Kp_factors,
            self.Kd_factors,
            self.motor_strength,
            self.contacts,
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # ---- 噪声 (height 段的 noise_scale=0,见 _get_noise_scale_vec / height_measurements=0) ----
        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            obs_buf_all = torch.stack([self.obs_history[i]
                                       for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        else:
            self.obs_buf = obs_now

    # ---------------- foothold penalty ----------------
    def _init_foot_sample_points(self):
        """Grid of (x,y,0) sample points across the foot sole, in the foot frame.
        Built once. Shape: (n_samples, 3)."""
        cfg = self.cfg.rewards
        xs = torch.linspace(-cfg.foot_length / 2, cfg.foot_length / 2,
                            cfg.foot_n_x, device=self.device)
        ys = torch.linspace(-cfg.foot_width / 2, cfg.foot_width / 2,
                            cfg.foot_n_y, device=self.device)
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        pts = torch.zeros(gx.numel(), 3, device=self.device)
        pts[:, 0] = gx.flatten()
        pts[:, 1] = gy.flatten()
        self.foot_sample_points = pts  # (S, 3)
        self.num_foot_samples = pts.shape[0]

    def _terrain_height_at(self, points_xy):
        """Terrain height lookup at arbitrary world XY.
        points_xy: (K, 2)  ->  heights: (K,).  Mirrors base _get_heights()."""
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(points_xy.shape[0], device=self.device)
        pts = points_xy + self.terrain.cfg.border_size
        pts = (pts / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(pts[:, 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(pts[:, 1], 0, self.height_samples.shape[1] - 2)
        # min over the neighbouring cells = conservative (matches _get_heights)
        h = torch.min(torch.min(self.height_samples[px, py],
                                self.height_samples[px + 1, py]),
                      self.height_samples[px, py + 1])
        return h * self.terrain.cfg.vertical_scale

    def _reward_foothold(self):
        # lazy one-time init (feet_pos etc. exist after _init_foot)
        if not hasattr(self, 'foot_sample_points'):
            self._init_foot_sample_points()

        E, F, S = self.num_envs, self.feet_num, self.num_foot_samples

        # rotate sole samples by each foot's yaw, translate to world
        quat = self.feet_quat.reshape(E * F, 4).unsqueeze(1).expand(-1, S, -1)  # (E*F, S, 4)
        pts = self.foot_sample_points.unsqueeze(0).expand(E * F, -1, -1)  # (E*F, S, 3)
        world = quat_apply_yaw(quat.reshape(-1, 4), pts.reshape(-1, 3))  # (E*F*S, 3)
        world = world.reshape(E, F, S, 3)
        world = world + self.feet_pos.unsqueeze(2)  # + foot xyz

        # terrain height under every sample
        terr = self._terrain_height_at(world[..., :2].reshape(-1, 2)).reshape(E, F, S)

        # 参考面 = 该脚自身采样点里最高的地形 = 它实际踩着的石块 / 横梁表面。
        # 用【相对高度】而非 feet_pos.z:feet_pos.z 是 ankle 关节原点,远高于脚底(~脚厚),
        # 平地站立时 foot_z - terr 恒 > ε,会把每个触地脚都判为悬空 → 逼出单腿跳。
        ref = terr.max(dim=-1, keepdim=True).values  # (E, F, 1) 支撑面
        # d_ij : 采样点地形比支撑面低多少;> ε 表示该点悬在石块之外(空洞上方)
        bad = ((ref - terr) > self.cfg.rewards.foothold_depth_tol).float()  # 1{...}

        # C_i * Σ_j 1{...}, summed over feet.  Sign comes from the config scale.
        Ci = self.contacts.float()  # (E, F)
        return (Ci * bad.sum(dim=-1)).sum(dim=-1)  # (E,)

    # ---------------- world-frame progress / heading (anti-detour) ----------------
    # tracking_lin_vel/ang_vel are computed in the robot's own base frame, so a
    # robot that turns away from an obstacle and keeps walking "forward" in its
    # new heading collects full reward -- nothing penalizes abandoning the
    # originally-commanded direction. These two terms anchor to the WORLD-frame
    # direction implied by each freshly-resampled local command (see
    # _resample_commands above) and reward actual progress/heading against that
    # fixed target, Extreme-Parkour-style (arXiv:2309.14341's world-frame
    # r_tracking = min(<v, d_hat>, v_cmd)), so detouring or retreating shows up
    # as reduced/negative reward instead of being reward-neutral.
    def _reward_world_progress(self):
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        world_vel = self.root_states[:, 7:9]  # world-frame xy velocity (unrotated)
        proj = torch.sum(world_vel * self.commands_world_dir, dim=1)
        rew = torch.clamp(proj, max=self.commands_world_speed)
        rew[self.standing_cmd] = 0.
        return rew

    def _reward_world_heading(self):
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_target = torch.atan2(self.commands_world_dir[:, 1], self.commands_world_dir[:, 0])
        heading_error = wrap_to_pi(heading_target - heading)
        rew = torch.exp(-torch.square(heading_error) * 2.0)
        rew[self.standing_cmd] = 0.
        return rew