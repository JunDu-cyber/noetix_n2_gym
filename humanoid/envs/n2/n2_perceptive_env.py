import torch
from isaacgym.torch_utils import quat_apply
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi
from humanoid.envs.n2.n2_10dof_env import N2_10dof_Env

class N2PerceptiveEnv(N2_10dof_Env):
    # ------------------------------------------------------------------
    # World-frame reference heading (anti-detour)
    # ------------------------------------------------------------------
    # Round-4 redesign. The previous version FROZE the world-frame command
    # direction at each resample and never rotated it again -- but
    # commands[:, 2] simultaneously asks for up to +-1 rad/s of yaw for 5-15 s
    # (resampling_time), so the frozen target drifted up to +-15 rad relative
    # to the robot *by design*. Measured consequence (logs/n2_perceptive/
    # 0724_17-08-04_): a Monte-Carlo robot that tracks its commands perfectly
    # and never detours scores world_progress 0.517 / world_heading 0.408,
    # and the trained policy scored 0.434 / 0.388 -- i.e. the terms were
    # saturated by the yaw command alone and carried no information about
    # detouring, while injecting a target the policy cannot observe (no
    # absolute yaw in obs, frame_stack covers only 0.2 s) at a combined scale
    # larger than tracking_lin_vel + tracking_ang_vel. That showed up as
    # value_function loss ~3-4x the flat baseline, mean_noise_std climbing
    # monotonically 1.0 -> 2.61, tracking rewards decaying and terrain_level
    # stalling.
    #
    # Now the reference yaw INTEGRATES the commanded yaw rate, so a robot that
    # obeys all three command channels scores full marks and only an
    # *uncommanded* yaw excursion (a detour turn) or a positional
    # sidestep/retreat loses reward -- the intended semantics. It also
    # penalises the *integral* of the yaw error, which is exactly what the
    # rate-based tracking_ang_vel cannot do: a 2 s 90-degree detour turn costs
    # a little rate error once, then collects full base-frame tracking_lin_vel
    # for the next 10 s.
    def _resample_commands(self, env_ids):
        """Seed the reference yaw to the robot's actual yaw whenever a new
        command is drawn, so each command segment starts with zero heading
        error and any drift accumulated under the previous command is
        forgiven.

        Reads the quaternion from root_states rather than self.base_quat on
        purpose: this is called from reset_idx AFTER _reset_root_states but
        BEFORE the "fix reset gravity bug" block that refreshes base_quat
        (legged_robot.py:219), so base_quat is still the pre-reset orientation
        at this point while root_states is already the new spawn."""
        if not hasattr(self, 'yaw_ref'):
            self._init_world_progress_buffers()
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        forward = quat_apply(self.root_states[env_ids, 3:7], self.forward_vec[env_ids])
        self.yaw_ref[env_ids] = torch.atan2(forward[:, 1], forward[:, 0])

    def _init_world_progress_buffers(self):
        self.yaw_ref = torch.zeros(self.num_envs, device=self.device)
        self.world_heading_err = torch.zeros(self.num_envs, device=self.device)
        self.commands_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self.commands_world_speed = torch.zeros(self.num_envs, device=self.device)
        self.world_progress_accum = torch.zeros(self.num_envs, device=self.device)

    def _post_physics_step_callback(self):
        # super() resamples any expired commands (seeding yaw_ref for those
        # envs) and refreshes standing_cmd; the world reference must be built
        # on top of that, and before check_termination/compute_reward, which
        # post_physics_step calls straight after this (legged_robot.py:113-117).
        super()._post_physics_step_callback()
        self._update_world_reference()

    def _update_world_reference(self):
        """Integrate the commanded yaw rate into yaw_ref, leak-clamp it to the
        robot's actual yaw, and rebuild the world-frame command direction."""
        if not hasattr(self, 'yaw_ref'):
            self._init_world_progress_buffers()

        self.yaw_ref += self.commands[:, 2] * self.dt

        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        # Leak clamp: bound how far the reference may run ahead of the robot.
        # Without it, a command the robot physically cannot track (a fall, or a
        # hard turn on stairs) lets yaw_ref race away at up to 1 rad/s for 15 s
        # and both terms collapse for reasons outside the policy's control --
        # pure return variance. max_err is deliberately pi/2 rather than
        # something tighter: at pi/2 a 90-degree detour turn still drives
        # world_progress to ~0 (cos(pi/2)) and world_heading to ~0.007, i.e.
        # the full anti-detour signal survives, and only *beyond* 90 degrees
        # does the penalty saturate -- which is fine, that is already maximal.
        max_err = self.cfg.rewards.world_heading_max_err
        self.world_heading_err = torch.clamp(wrap_to_pi(self.yaw_ref - yaw), -max_err, max_err)
        self.yaw_ref = yaw + self.world_heading_err

        # world-frame direction of the base-frame linear velocity command,
        # rotated by the REFERENCE yaw (not the actual yaw) -- that difference
        # is the whole anti-detour signal.
        c, s = torch.cos(self.yaw_ref), torch.sin(self.yaw_ref)
        vx, vy = self.commands[:, 0], self.commands[:, 1]
        world_vel_cmd = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=1)
        self.commands_world_speed = torch.norm(world_vel_cmd, dim=1)
        self.commands_world_dir = world_vel_cmd / self.commands_world_speed.clamp(min=1e-6).unsqueeze(1)

        # Directional progress for the terrain curriculum, accumulated
        # incrementally so it is exact regardless of how many command segments
        # an episode spans. This replaces the reference-position/segment
        # bookkeeping of rounds 2-3, whose whole bug class (progress made under
        # an earlier command being erased by a later one) simply cannot occur
        # when each step is credited against the direction in force that step.
        self.world_progress_accum += torch.sum(
            self.root_states[:, 7:9] * self.commands_world_dir, dim=1) * self.dt

    def reset_idx(self, env_ids):
        # super() runs _update_terrain_curriculum first, which consumes
        # world_progress_accum for the episode that just ended, so only zero
        # it afterwards.
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, 'world_progress_accum'):
            self.world_progress_accum[env_ids] = 0.

    def _update_terrain_curriculum(self, env_ids):
        """Directional variant of the base class's radial-distance curriculum
        (legged_robot.py:513). The base version levels up on ANY net
        displacement from spawn, which a centrally-symmetric obstacle (or just
        circling/retreating) satisfies as validly as actually crossing it --
        confirmed in logs/n2_perceptive/0724_11-26-53_, where terrain_level
        climbed to ~4.9 while rew_stumble/rew_collision stayed near zero (the
        robot was rarely making real contact with the stairs at all). Uses the
        per-step accumulated projection onto commands_world_dir instead, so
        credit only accrues for progress in the direction actually asked for.
        Only overridden here, not in the shared base class, so n2_10dof/n2
        (no world-frame buffers) keep the original radial behaviour."""
        if not self.init_done:
            return
        if not hasattr(self, 'world_progress_accum'):
            super()._update_terrain_curriculum(env_ids)
            return
        progress = self.world_progress_accum[env_ids]

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
    # See the _update_world_reference block at the top of this class for why
    # the reference yaw integrates the commanded yaw rate instead of being
    # frozen at resample, and for the measured evidence that the frozen
    # version carried no anti-detour information at all.
    def _reward_world_progress(self):
        """Actual world-frame velocity projected on the commanded world
        direction, Extreme-Parkour-style (arXiv:2309.14341's
        r_tracking = min(<v, d_hat>, v_cmd)). A robot obeying all three command
        channels keeps its heading aligned with yaw_ref and scores the full
        commanded speed; one that turns away to skirt an obstacle, or sidesteps
        or retreats, loses the projection."""
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        world_vel = self.root_states[:, 7:9]  # world-frame xy velocity (unrotated)
        proj = torch.sum(world_vel * self.commands_world_dir, dim=1)
        # Symmetric clamp: commands_world_speed bounds the positive side, but
        # proj itself is unbounded -- a fall/push/stumble can spike world_vel
        # in the wrong direction with no floor, and at a large scale that
        # single step dwarfs the rest of the stack (a measured -20+ single-step
        # contribution at scale 8.0 diverged a run, noise_std 1.0 -> 21.0).
        rew = torch.clamp(proj, min=-self.commands_world_speed, max=self.commands_world_speed)
        rew[self.standing_cmd] = 0.
        return rew

    def _reward_world_heading(self):
        """Penalise the *accumulated* yaw error against the commanded yaw rate.
        Target is yaw_ref, NOT the direction of the linear velocity command:
        that earlier choice demanded a mean 90-degree (median 90-degree)
        instantaneous body turn over the real command distribution -- 46% of
        commands asked for >90 degrees and any vx<0 command asked for ~180 --
        which fought tracking_lin_vel and world_progress simultaneously.
        Extreme Parkour has no such conflict only because its commands are
        always goal-directed/forward-facing."""
        if not hasattr(self, 'world_heading_err'):
            return torch.zeros(self.num_envs, device=self.device)
        rew = torch.exp(-torch.square(self.world_heading_err) * 2.0)
        rew[self.standing_cmd] = 0.
        return rew

    # ---------------- bounded stand-still penalty ----------------
    def _reward_stand_still(self):
        """De-weighted, capped version of N2_10dof_Env._reward_stand_still.

        The inherited term is `sum|dof_pos-default| + sum(dof_vel^2)`: an L1
        pose term plus an L2 joint-velocity term that is quadratic and
        unbounded. Only standing_cmd envs are scored (~20% of envs, since
        n2_10dof_env.py:134 zeroes every command on 20% of resamples).

        Measured under real training conditions (sampled actions, i.e. WITH
        the policy's exploration noise -- the dominant driver of dof_vel, and
        the thing an inference-mode probe misses entirely) on the
        0723_19-51-09_ checkpoint:

          standing envs   raw mean 88.8, median 43.5, p95 338, max 3917
                          per-step total reward BEFORE clipping: mean -0.162
                          clipped to 0 by only_positive_rewards: 75.3%
          moving envs     per-step total mean -0.009, clipped: 17.0%

        So standing envs spent three quarters of their steps pinned at exactly
        0 total reward. Zero reward variation means zero advantage, so the
        standing posture was never actually trained -- which is visible in
        deployment as a robot that shakes badly while commanded to stand and
        goes quiet the moment a velocity command arrives. Note this was
        measured on a run WITHOUT the world rewards, so the dead zone is not
        caused by them; they deepen it (rew_stand_still scales with the world
        reward scale: 1.0/0.5 -> -2.9, 3.5/1.5 -> -4.6, 5.0/2.5 -> -9.0)
        because higher noise_std means more action noise means quadratically
        more dof_vel^2.

        Fix: de-weight the quadratic term so the standing state sits back in
        positive territory and gets a gradient again, and keep a cap as an
        outlier guard on top. A cap ALONE does not work -- capping raw at 80
        still left 75% of standing steps clipped, because the problem is the
        typical value (median 43.5), not just the tail. De-weighting is also
        preferable to simply lowering the scale: it keeps a non-zero gradient
        on dof_vel everywhere, whereas above a hard cap that gradient is
        exactly zero, removing the very signal that is supposed to quiet the
        joints down.

        Overridden here rather than in N2_10dof_Env so the blind n2_10dof/n2
        tasks are untouched."""
        rew = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        sc = self.standing_cmd
        rew[sc] = (torch.sum(torch.abs(self.dof_pos[sc] - self.default_dof_pos), dim=1)
                   + self.cfg.rewards.stand_still_vel_weight
                   * torch.sum(torch.square(self.dof_vel[sc]), dim=1))
        return torch.clamp(rew, max=self.cfg.rewards.stand_still_max)
