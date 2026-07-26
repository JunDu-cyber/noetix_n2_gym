import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float
from humanoid.utils.math import quat_apply_yaw, wrap_to_pi
from humanoid.envs.n2.n2_10dof_env import N2_10dof_Env

class N2PerceptiveEnv(N2_10dof_Env):
    # ---- 世界系参考朝向 (反绕路) ----
    # 参考 yaw 必须【积分】指令偏航率，不能在 resample 时冻结：commands[:,2] 可以
    # 在一段 5~15s 的指令内要求最多 ±1 rad/s，冻结的目标会按设计漂走 ±15 rad，
    # 于是这两项被偏航指令本身饱和、不含任何绕路信息(实测完美服从的机器人得分与
    # 训练策略相同)，反而在策略观测不到的维度上注入方差。
    # 积分之后：服从全部三个指令通道即得满分，只有【未被指令的】偏航偏移(绕路转弯)
    # 或位置上的侧移/后退才扣分。它罚的是偏航误差的积分——正是速率型 tracking_ang_vel
    # 罚不到的东西(转身 2s 只付一次速率误差，之后 10s 照拿满额 tracking_lin_vel)。
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

        # 楼梯列把指令限制到该地形唯一的可穿越轴 (+x)，等同 Extreme Parkour 在定向
        # 地形上的做法。这是下面课程改动(只按 +x 计分)的前提：vx~U(-0.8,0.8) 关于 0
        # 对称，完美服从的机器人整段净 x 位移期望为 0，不先把指令偏向 +x，按穿越轴
        # 计分只会让楼梯列永久降级。wz 也必须置零，否则 yaw_ref 积分它、
        # commands_world_dir 会在段内转离 +x。只影响 index 7/8。
        if getattr(self.cfg.commands, 'stairs_forward_only', False):
            on_stairs = self._stairs_env_mask()[env_ids]
            if torch.any(on_stairs):
                ids = env_ids[on_stairs]
                # 站立指令保持站立（不把它强行变成前进），与基类语义一致
                moving = torch.norm(self.commands[ids, :3], dim=1) > self.min_cmd_vel
                ids = ids[moving]
                if len(ids) > 0:
                    # 必须是重采样而不是 abs().clamp(min=lo)：clamp 会在下限堆出
                    # 质量点(37.5% 的 |vx| 落在 0.3 以下、全被压到 0.3)，等于从第 0
                    # 轮就强制 30% 的环境快走，实测 ep_len 5.9 vs 9.5。
                    lo = self.cfg.commands.stairs_min_vx
                    hi = max(self.command_ranges["lin_vel_x"][1], lo)
                    self.commands[ids, 0] = torch_rand_float(
                        lo, hi, (len(ids), 1), device=self.device).squeeze(1)
                    self.commands[ids, 1] = 0.
                    self.commands[ids, 2] = 0.

        forward = quat_apply(self.root_states[env_ids, 3:7], self.forward_vec[env_ids])
        self.yaw_ref[env_ids] = torch.atan2(forward[:, 1], forward[:, 0])

    def _stairs_env_mask(self):
        """每个环境是否位于直行楼梯列（terrain_proportions 的 index 7/8）。

        非课程地形（plane / selected，没有 terrain_types）时全 False，B/C 自动失效，
        play 和盲策略路径都不受影响。"""
        if not hasattr(self, 'terrain_types'):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, 'stairs_tile'):
            self.stairs_tile = self._stairs_tile_mask()
        return self.stairs_tile[self.terrain_types]

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
        # 泄漏钳制：限制参考 yaw 领先机器人的幅度。否则一条物理上跟不上的指令
        # (摔倒、楼梯上急转)会让 yaw_ref 以 1 rad/s 跑 15s，两项因策略无法控制的
        # 原因崩掉，纯粹是回报方差。取 π/2 而非更紧：在 π/2 处 90° 绕路转弯仍把
        # world_progress 压到 ~0、world_heading 压到 ~0.007，反绕路信号完整保留。
        max_err = self.cfg.rewards.world_heading_max_err
        self.world_heading_err = torch.clamp(wrap_to_pi(self.yaw_ref - yaw), -max_err, max_err)
        self.yaw_ref = yaw + self.world_heading_err

        # 用【参考】yaw 而非实际 yaw 旋转指令——两者之差就是全部的反绕路信号。
        c, s = torch.cos(self.yaw_ref), torch.sin(self.yaw_ref)
        vx, vy = self.commands[:, 0], self.commands[:, 1]
        world_vel_cmd = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=1)
        self.commands_world_speed = torch.norm(world_vel_cmd, dim=1)
        self.commands_world_dir = world_vel_cmd / self.commands_world_speed.clamp(min=1e-6).unsqueeze(1)

        # 课程用的方向性进展。逐步累加(而非记录 episode 起点位移)：每步都记在当步
        # 生效的方向上，跨指令段自动正确——否则一个 episode 横跨 2+ 个指令方向时，
        # 前一段的进展会被后一段的方向抹掉。
        prog = torch.sum(self.root_states[:, 7:9] * self.commands_world_dir, dim=1)

        # 楼梯列只按世界系 +x 计分：地形沿 y 恒定，"沿 commands_world_dir 的进展"
        # 不等于"穿越"——沿 y 走 3 秒(一级台阶都不碰)就能满足升级阈值。这挡住的是
        # 机器人物理上转向的情形：即使 wz 指令为 0 实际偏航仍会漂移，yaw_ref 的泄漏
        # 钳制被拖走后 commands_world_dir 会转离 +x，那正是绕路本身。
        # 对称地形(金字塔/方块)不替换：那里任意方向的进展本就等于穿越等高线。
        on_stairs = self._stairs_env_mask()
        prog = torch.where(on_stairs, self.root_states[:, 7], prog)

        self.world_progress_accum += prog * self.dt

    def reset_idx(self, env_ids):
        # super() runs _update_terrain_curriculum first, which consumes
        # world_progress_accum for the episode that just ended, so only zero
        # it afterwards.
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and hasattr(self, 'world_progress_accum'):
            self.world_progress_accum[env_ids] = 0.

    def _stairs_tile_mask(self):
        """哪些地形列是直行楼梯（terrain_proportions 的 index 7/8 分支）。

        复刻 Terrain.curiculum() 的 choice=j/num_cols+0.001 与 make_terrain 的 elif
        链，只在 curriculum=True 的确定性网格下成立（训练就是这么跑的）。
        """
        props = self.cfg.terrain.terrain_proportions
        cum = [float(np.sum(props[:i + 1])) for i in range(len(props))]
        n = self.cfg.terrain.num_cols
        mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        for j in range(n):
            choice = j / n + 0.001
            idx = next((k for k, c in enumerate(cum) if choice < c), -1)
            mask[j] = idx in (7, 8)
        return mask

    def _reset_root_states(self, env_ids):
        """把直行楼梯格的出生点从块中心挪到 -x 端的底部平台。

        directional_stairs 让楼梯从 -x 端底部平台贯穿整块地形往 +x 升/降。基类把
        机器人放在块中心（env_origins）——那是半山腰、可爬长度只剩一半，而且基类的
        env_origin_z = max(中心±1m 窗口) 在单调楼梯上取的是前方 1m 处的最高台阶，
        实测会让出生点悬空 0.2~0.8m（台阶越高越严重），每次 reset 都在自由落体。
        这里对楼梯列把出生点移到底部平台中央：
          x: env_origin_x - env_length/2 + platform/2，抖动收窄到 ±(platform*0.4)，
             免得越过 -x 端背墙(上一级楼梯的顶)或直接生到台阶上。
          z: 底部平台恒为 0 高度，改成 base_init 站立高度 + 0.05，消除上述悬空。
        y 保持基类的 ±1m（楼梯沿 y 恒高，任意 y 出生等价）。其他地形不受影响。
        """
        super()._reset_root_states(env_ids)
        if not self.custom_origins or len(env_ids) == 0:
            return
        if not hasattr(self, 'stairs_tile'):
            self.stairs_tile = self._stairs_tile_mask()
        on_stairs = self.stairs_tile[self.terrain_types[env_ids]]
        if not torch.any(on_stairs):
            return
        ids = env_ids[on_stairs]
        plat = getattr(self.cfg.terrain, 'stairs_platform_size', 1.5)
        base_x = self.env_origins[ids, 0] - self.terrain.env_length / 2. + plat / 2.
        jit = min(plat * 0.4, 0.5)
        self.root_states[ids, 0] = base_x + torch_rand_float(
            -jit, jit, (len(ids), 1), device=self.device).squeeze(1)
        # 底部平台高度=0，覆盖掉基类按块中心高度设的 root_z
        self.root_states[ids, 2] = self.base_init_state[2] + 0.05
        ids32 = ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

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

        # up-distance: base uses env_length/2 (2m). With center spawn on stairs that
        # means climbing the whole upper half-tile to level up -- too steep for early
        # stairs, which pins the stairs columns at level 0. Configurable, default 1.5m.
        up_dist = getattr(self.cfg.terrain, 'curriculum_up_distance', self.terrain.env_length / 2)
        move_up = progress > up_dist
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
        # d_ij : 采样点地形比支撑面低多少 = 该点悬在支撑面之外的深度(m)
        d = (ref - terr).clamp(min=0.)  # (E, F, S)

        # raw = 每只触地脚 (1 - exp(-k*d̄)) 的均值 ∈ [0,1]，必须【有界且平滑】
        # (同类工作 Limx Oli 的 feet_stair_flat=exp(-4*r_D) 也是指数形式)。
        # 原实现是离散计数 sum(1{d > 4cm})：中位数恒为 0(一半以上的步没有梯度)，
        # 上界 12 使单步最坏惩罚 -3.5，而整个正项栈才 ~1.5 —— 正是尖峰致发散的通道。
        # d̄ 取每只脚采样点的均值而非最大值，避免单个边缘点主导整只脚的评分。
        k = self.cfg.rewards.foothold_flat_k
        per_foot = 1.0 - torch.exp(-k * d.mean(dim=-1))  # (E, F) in [0, 1]
        Ci = self.contacts.float()  # (E, F)
        n_contact = Ci.sum(dim=-1).clamp(min=1.0)
        # 腾空(无触地脚)时为 0：既不奖励也不惩罚，避免把"飞行相"变成可刷的状态。
        return (Ci * per_foot).sum(dim=-1) / n_contact  # (E,) in [0, 1]

    # ---- 世界系 progress / heading (反绕路)，设计理由见类顶部注释 ----
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
        # 必须除以指令速度(同 EP 的 _reward_tracking_goal_vel)：返回原始投影会让
        # 该项速度相关——服从 0.3m/s 指令得 0.3、服从 0.8m/s 得 0.8，结构性地惩罚
        # 慢速，而慢正是小心爬楼需要的。归一化后任意指令速度下满分都是 1.0。
        # 分母下限防 wz 主导的指令(min_cmd_vel 只约束三维模长，|v_xy| 可接近 0)。
        # 对称 [-1,1] 钳制是尖峰防护：proj 下无界，一次摔倒/推挤反向打飞 world_vel
        # 就曾把 noise_std 从 1.0 推到 21.0。
        denom = self.commands_world_speed.clamp(min=self.cfg.rewards.world_progress_min_speed)
        rew = torch.clamp(proj / denom, min=-1.0, max=1.0)
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

    # ---------------- bounded foot-impact penalty ----------------
    def _reward_feet_contact_forces(self):
        """有界版的接触力惩罚，替代基类的 sum(clip(|F| - max_contact_force, 0, inf))。

        基类那版有两个问题，实测都在咬人：
        1) 阈值 300N 低于本机器人自重（33.2kg = 325N）。单脚支撑期——正常步态里
           一半的时间——那只脚就承受全部体重，于是"仅仅是正常走路"都在持续扣分。
           这解释了此前对照实验里它为何是**地形无关**的（平地 -4.85 / 楼梯 -5.0）：
           它根本不是在惩罚"踩楼梯太重"，而是一笔恒定的步态税。
        2) 上不封顶。摔倒砸地时 |F| 轻易上千牛，单步惩罚可以盖过整个正项栈。这与
           已经修过的 world_progress（scale 8.0 把 noise_std 从 1.0 推到 21.0）和
           foothold（离散计数上界 12、单步 -3.5）是同一条尖峰通道。

        实测发散 run 0726_10-08-30_：本项 -5.46/秒，而全部正项合计只有 +0.41/秒。
        因为 only_positive_rewards=True 会把每步总奖励截断到 0，**每一步都被截断**，
        奖励梯度完全消失、只剩熵项，于是 noise_std 单调爬到 2.6，机器人 400 轮都没
        学会站住（episode 长度 ~10 步）。同期健康 run 的净值是 -0.34/秒，勉强贴在
        截断边界上——可见这个奖励栈本来就悬在悬崖边，任何扰动都会把它推下去。
        """
        excess = (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                  - self.cfg.rewards.max_contact_force).clip(min=0.)
        return excess.clip(max=self.cfg.rewards.feet_contact_force_max_excess).sum(dim=1)

    # ---------------- anti-freeze (break the "stand still on stairs" optimum) ----------------
    def _reward_anti_freeze(self):
        """Positive reward for command-aligned forward world-speed, saturating at
        `rewards.anti_freeze_speed`.

        Motivation: with only_positive_rewards=True the summed step reward is
        clipped at 0, so a robot commanded to move but standing still sits at
        exactly 0 -- and a climbing attempt that stumbles also clips to 0. That
        erases the gradient favouring "move" over "freeze" on risky terrain,
        which is why the policy climbs in Isaac yet freezes on the *same* stairs
        in MuJoCo (verified: spawned on the stairs, commanded forward, it stands
        with world-vx approx 0 for 30 s). A *penalty* cannot fix this -- it is
        clipped away with everything else. So this is a POSITIVE term instead.

        It saturates at anti_freeze_speed (a few cm/s), so it is NOT a speed race
        and does not fight tracking_lin_vel: the entire reward is earned crossing
        from 0 to anti_freeze_speed, i.e. the steepest gradient sits exactly at
        the freeze point. Standing earns 0, any real forward motion earns up to
        1, lifting "move" above "freeze" even after the only_positive_rewards
        clip. Uses the same world-frame velocity / commands_world_dir projection
        as _reward_world_progress (so retreat/detour earns nothing here either).
        Gated off for standing_cmd envs -- they are meant to hold still, so this
        never fights _reward_stand_still."""
        if not hasattr(self, 'commands_world_dir'):
            return torch.zeros(self.num_envs, device=self.device)
        fwd_speed = torch.sum(self.root_states[:, 7:9] * self.commands_world_dir, dim=1)
        rew = torch.clamp(fwd_speed / self.cfg.rewards.anti_freeze_speed, min=0.0, max=1.0)
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
