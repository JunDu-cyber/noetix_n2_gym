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
        observed command, so it carries no sim2sim deployment burden.

        Also maintains world_progress_accum/world_progress_ref_pos for
        _update_terrain_curriculum: commands resample every 5-15s
        (resampling_time) while an episode can run up to 20s
        (episode_length_s), so most episodes span 2+ different commanded
        directions. commands_world_dir gets overwritten every resample, so a
        robot correctly walking north under command 1 then east under
        command 2 would have that northward progress erased if curriculum
        leveling only looked at total spawn-to-now displacement against the
        FINAL direction -- this is what caused terrain_level to climb fine
        early (episodes too short to hit a resample) and then plateau once
        episodes got long enough to commonly span multiple commands. Fix:
        fold each completed segment's progress into a running per-episode
        accumulator as its direction is about to be replaced, so multi-
        segment episodes get credited fairly.

        This is called from two places with different position semantics:
        the periodic per-step resample in _post_physics_step_callback (root
        state hasn't moved since last resample -- safe to measure the
        just-finished segment here), and reset_idx (root state has ALREADY
        been reset to the new spawn point by the time this runs, since
        _resample_commands is called after _reset_root_states -- measuring
        anything here would compare the new spawn against the old episode's
        reference point, which is meaningless). reset_buf[env_ids] tells
        them apart: check_termination sets it before reset_idx is entered,
        and it's stale (from last step, effectively 0 for a continuing
        episode) during the periodic path since that runs before this
        step's check_termination. The reset case's final segment is instead
        folded in by _update_terrain_curriculum, which runs before
        _reset_root_states -- see there."""
        if not hasattr(self, 'commands_world_dir'):
            self._init_world_progress_buffers()
        if len(env_ids) == 0:
            super()._resample_commands(env_ids)
            return
        is_reset = self.reset_buf[env_ids].bool()
        continuing = env_ids[~is_reset]
        if len(continuing) > 0:
            seg_disp = self.root_states[continuing, :2] - self.world_progress_ref_pos[continuing]
            self.world_progress_accum[continuing] += torch.sum(seg_disp * self.commands_world_dir[continuing], dim=1)
        reset_ids = env_ids[is_reset]
        if len(reset_ids) > 0:
            # _update_terrain_curriculum already folded the final segment of
            # the episode that just ended into a local total for these envs
            self.world_progress_accum[reset_ids] = 0.

        super()._resample_commands(env_ids)
        local_vel = torch.zeros(len(env_ids), 3, device=self.device)
        local_vel[:, :2] = self.commands[env_ids, :2]
        world_vel = quat_apply_yaw(self.base_quat[env_ids], local_vel)[:, :2]
        speed = torch.norm(world_vel, dim=1)
        self.commands_world_speed[env_ids] = speed
        self.commands_world_dir[env_ids] = world_vel / speed.clamp(min=1e-6).unsqueeze(1)
        self.world_progress_ref_pos[env_ids] = self.root_states[env_ids, :2].clone()

    def _init_world_progress_buffers(self):
        self.commands_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self.commands_world_speed = torch.zeros(self.num_envs, device=self.device)
        self.world_progress_accum = torch.zeros(self.num_envs, device=self.device)
        self.world_progress_ref_pos = torch.zeros(self.num_envs, 2, device=self.device)

    def _update_terrain_curriculum(self, env_ids):
        """Directional variant of the base class's radial-distance curriculum
        (legged_robot.py:513). The base version levels up on ANY net
        displacement from spawn, which a centrally-symmetric obstacle (or
        just circling/retreating) satisfies as validly as actually crossing
        it -- confirmed in logs/n2_perceptive/0724_11-26-53_, where
        terrain_level climbed to ~4.9 while rew_stumble/rew_collision stayed
        near zero (i.e. the robot was rarely attempting real contact with the
        stairs at all). This projects displacement onto commands_world_dir
        (the frozen world-frame direction of each command segment, see
        _resample_commands above) instead of taking radial magnitude, so
        credit only accrues for progress in the direction actually asked
        for -- including correctly demoting a robot that retreats. Only
        overridden here, not in the shared base class, so n2_10dof/n2 (no
        world-frame buffers) keep the original radial-distance behavior
        unaffected.

        Runs before _reset_dofs/_reset_root_states (see reset_idx), so
        root_states here is still the position where the episode actually
        ended -- folds that final segment's progress into
        world_progress_accum (built up over any earlier segments this
        episode by _resample_commands) to get the episode's total directional
        progress, without needing root_states from after the upcoming reset."""
        if not self.init_done:
            return
        if not hasattr(self, 'commands_world_dir'):
            super()._update_terrain_curriculum(env_ids)
            return
        seg_disp = self.root_states[env_ids, :2] - self.world_progress_ref_pos[env_ids]
        seg_progress = torch.sum(seg_disp * self.commands_world_dir[env_ids], dim=1)
        progress = self.world_progress_accum[env_ids] + seg_progress

        move_up = progress > self.terrain.env_length / 2
        move_down = (progress < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

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
        # Symmetric clamp: commands_world_speed only bounds the *positive* side
        # (<=~0.94 m/s), but proj itself is unbounded -- a fall/push/stumble can
        # spike world_vel in the wrong direction with no floor, and at a large
        # scale that single step can dwarf the rest of the reward stack (a
        # measured -20+ single-step contribution at scale=8.0), which showed up
        # as PPO divergence (noise_std 1.0->21.0 over one run) rather than a
        # useful anti-retreat signal. Clamp both sides so retreat still costs
        # reward without being able to produce unbounded outliers.
        rew = torch.clamp(proj, min=-self.commands_world_speed, max=self.commands_world_speed)
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