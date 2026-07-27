import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float
from isaacgym import gymapi, gymutil
from humanoid.utils.math import wrap_to_pi
from humanoid.utils.terrain import ParkourTerrain
from humanoid.envs.n2.n2_perceptive_env import N2PerceptiveEnv


class N2ParkourEnv(N2PerceptiveEnv):
    """Extreme Parkour(arXiv:2309.14341)架构的复刻。

    与 n2_perceptive 的根本区别是【目标从"速度指令方向"换成了"goal 路点位置"】。
    这解决的是我们反复撞上的那个结构性错配：定向地形要求朝 +x 穿越，而全向随机速度
    指令有一半时间指向别处，于是课程能靠沿 y 走刷级、策略也学不会爬。EP 不存在这个
    问题，因为它压根不发横移/偏航指令，且奖励锚在"必须踩上去"的 goal 上。

    三个要件，逐条对齐 EP 的源码：
      1) 指令只有前进速度：lin_vel_y=[0,0]、ang_vel_yaw=[0,0]（EP 的 config 就是这么写的）。
      2) 奖励锚定 goal 位置而非方向：
           tracking_goal_vel = min(<d_hat, v_world>, vx_cmd) / vx_cmd
           tracking_yaw      = exp(-|wrap(atan2(d) - yaw)|)
         其中 d = cur_goal - root_xy 是【位置差】，会随机器人偏移而转向——这正是
         我们此前用 commands_world_dir（只有方向、不随位置更新）拿不到的那部分信号。
      3) goal 落在障碍上，地形是"中央通道 + y 向 pad"，横向绕行在几何上被堵死。

    goal 是位置目标，必须可观测，否则就是 Round-4 那个"目标不在观测里"的老坑。
    这里把 [cos Δψ, sin Δψ]（Δψ = 到当前 goal 的方位角与自身朝向之差）加进单帧观测，
    num_single_obs 因此 135 -> 137，对应 EP 的 delta-yaw-to-goal 观测。
    """

    # ---------------- terrain / goals ----------------
    def _create_terrain_impl(self):
        """基类在 create_sim 里硬编码了 HumanoidTerrain，这里换成带 goal 的 ParkourTerrain。"""
        self.terrain = ParkourTerrain(self.cfg.terrain, self.num_envs)

    def create_sim(self):
        # 复刻基类 create_sim，但把地形类换掉。只在 trimesh/heightfield 下有意义。
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self._create_terrain_impl()
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _check_goal_reach_consistency(self):
        """goal_reach_dist 必须小于最小 goal 间距，否则指针连跳、课程虚高。

        这两个参数分处 rewards 和 terrain 两段配置，耦合关系不写下来就会被忘掉——
        这次就是：goal_reach_dist 从 0.5 调到 0.4 时没人注意到 parkour_x_range
        的下限只有 0.2m，结果 78% 的相邻 goal 落在到达半径内，课程一路虚高到
        terrain_level 6.3 而策略在 8.8cm 台阶只有 11% 存活。
        """
        # 最小间距是【所有启用地形类型】里最小的那个，不能只看台阶——多地形混训后
        # 台阶不一定还是最紧的那一种，写死 parkour_x_range 会漏掉新类型。
        gap = self.terrain.min_goal_spacing()
        reach = self.cfg.rewards.goal_reach_dist
        if gap > 0 and reach >= gap:
            raise ValueError(
                "goal_reach_dist=%.3f 必须 < 各地形类型的最小 goal 间距 %.3f，"
                "否则相邻 goal 落在到达半径内、指针连跳导致课程虚高" % (reach, gap))

    def _reward_stumble(self):
        """摆动脚撞到竖直面的惩罚：连续量 + 只算摆动相。

        基类版本(legged_robot.py:1159)是二值判据 any(|Fxy| > 5|Fz|)，两个毛病：
          1) 轻蹭和狠踢同价，梯度不带强度信息；
          2) 它是【比值】判据，摆动脚只要分担了一点体重比值就不成立，会漏掉。
        实测(model_999，上楼梯 lvl5)：触发率 2.61%，累计只占正奖励栈的 0.82%，
        所以"加了 stumble 还是踢台阶"不是权重不够，是这一项根本没有分辨力。

        阈值全部取自实测接触力分布(自重 327N)：
          支撑相 Fz 中位数 207.6N、p90 381N；摆动相 Fz p90 只有 100.7N
            -> stance_force=80N 能干净地把"还没承重的脚"分出来。
          摆动相 Fxy p90=16.8N(正常噪声)、p99=265N(真撞上)
            -> min_force=20N 滤掉噪声，ref_force=200N 做归一化。
        归一化到单脚 [0,1]、两脚合计 [0,2]，与原来的二值量纲相当，所以 scale 不用重调；
        同时天然封顶，不会像 world_progress 那次被单步尖峰把 PPO 的价值函数打飞。
        """
        c = self.cfg.rewards
        f = self.contact_forces[:, self.feet_indices, :]
        fxy = torch.norm(f[:, :, :2], dim=2)
        fz = torch.abs(f[:, :, 2])
        swing = fz < c.stumble_stance_force              # 脚还没真正承重
        hit = ((fxy - c.stumble_min_force).clip(min=0.) / c.stumble_ref_force).clip(max=1.)
        return torch.sum(hit * swing.float(), dim=1)

    def _check_spawn_clearance(self, max_drop=0.15, samples=64):
        """出生点抖动之后必须仍落在实地上。

        goals[0] 由地形函数决定、抖动幅度由 terrain 配置决定，两者分处不同文件，改一边
        很容易忘掉另一边——踏石就踩过：goals[0] 放在平台边缘(离末端 0.1m)，配 ±0.3m 抖动
        后 6.7% 的出生点直接落到坑上、最深掉 0.52m，而机器人还是按平台高度摆的，等于每
        次 reset 都从半空掉进沟里。这里在构造时把这个跨段耦合钉死。
        """
        jit = getattr(self.cfg.terrain, 'parkour_spawn_jitter', 0.0)
        if jit <= 0 or not hasattr(self.terrain, 'height_field_raw'):
            return
        hs = self.cfg.terrain.horizontal_scale
        vs = self.cfg.terrain.vertical_scale
        border = int(self.cfg.terrain.border_size / hs)
        hf = self.terrain.height_field_raw
        g0 = self.terrain.goals[:, :, 0, :]                      # (rows, cols, 3)
        # 角点最危险，直接取抖动方框的边界网格
        off = np.linspace(-jit, jit, int(np.sqrt(samples)))
        dx, dy = np.meshgrid(off, off, indexing='ij')
        px = np.clip(((g0[..., None, None, 0] + dx) / hs).astype(int) + border,
                     0, hf.shape[0] - 1)
        py = np.clip(((g0[..., None, None, 1] + dy) / hs).astype(int) + border,
                     0, hf.shape[1] - 1)
        drop = hf[px, py] * vs - g0[..., None, None, 2]
        worst = float(drop.min())
        if worst < -max_drop:
            i, j = np.unravel_index(np.argmin(drop.min(axis=(2, 3))), drop.shape[:2])
            raise ValueError(
                "出生点抖动 ±%.2fm 会把机器人扔到比 goals[0] 低 %.2fm 的地方"
                "(最差在 row %d / col %d)。把该地形的 goals[0] 往平台内挪，"
                "或调小 terrain.parkour_spawn_jitter。" % (jit, -worst, i, j))

    def _init_goal_buffers(self):
        self._check_goal_reach_consistency()
        self._check_spawn_clearance()
        # (num_rows, num_cols, num_goals, 3) 的世界坐标路点表
        self.terrain_goals = torch.tensor(self.terrain.goals, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.num_goals = self.terrain_goals.shape[2]
        # 见 reset_idx：goals[0] 是出生点，第一个真正的目标是 goals[1]
        self.cur_goal_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.reached_goals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_delta_yaw = torch.zeros(self.num_envs, device=self.device)

    def _env_goals(self):
        """当前每个环境所在地形块的 goal 表 -> (N, num_goals, 3)。"""
        return self.terrain_goals[self.terrain_levels, self.terrain_types]

    def _cur_goal(self):
        g = self._env_goals()
        return g[torch.arange(self.num_envs, device=self.device), self.cur_goal_idx]

    def _update_goals(self):
        """推进 goal 指针：走到当前 goal 的 reach 半径内就切下一个。"""
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        cur = self._cur_goal()
        self.target_pos_rel = cur[:, :2] - self.root_states[:, :2]
        dist = torch.norm(self.target_pos_rel, dim=1)
        reach = self.cfg.rewards.goal_reach_dist

        # 两个推进条件，缺一不可：
        # (a) 进到 reach 半径内。半径【必须小于最小 goal 间距】(见 config 里的
        #     assert)，否则站在一个 goal 上时下一个已在圈内、指针连跳，机器人挪到
        #     第一级台阶附近就能连拿 5 个 goal 升级，根本不用逐级爬。
        # (b) 已【越过】该 goal 且横向没跑出通道。只把半径改小会引入新故障：机器人
        #     可能擦过 goal 而没进圈，goal 留在身后，target_pos_rel 掉头指向后方，
        #     tracking_goal_vel 变负、tracking_yaw 要求它转身回去，通道就走不下去。
        #     "越过"要求真实 +x 位移，所以它触发的连跳是合法的；横向门限防止机器人
        #     绕到通道外的低地上沿 y 平移把 goal 一路"越过"。
        passed = (self.target_pos_rel[:, 0] < 0) & \
                 (self.target_pos_rel[:, 1].abs() < self.cfg.rewards.goal_pass_lateral_tol)
        advance = ((dist < reach) | passed) & (self.cur_goal_idx < self.num_goals - 1)
        self.cur_goal_idx[advance] += 1
        self.reached_goals[advance] += 1
        # 指针推进后重新取，保证 reward/obs 用的是同一步的目标
        cur = self._cur_goal()
        self.target_pos_rel = cur[:, :2] - self.root_states[:, :2]
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        target_yaw = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
        self.goal_delta_yaw = wrap_to_pi(target_yaw - yaw)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_goals()

    # ---------------- commands: 只发前进速度 ----------------
    def _resample_commands(self, env_ids):
        """EP 的指令空间只有 vx。这里在基类采样之后把 vy/wz 清零、vx 取正。

        注意与 n2_perceptive 的 stairs_forward_only 不同：那边是在【部分地形列】上
        改指令，结果和 _reward_default_joint_pos 的门控（有横移/偏航指令才白送满分）
        打架；这里是【整个任务】都不发横移/偏航，config 里把 lin_vel_y / ang_vel_yaw
        的范围直接设成 0，门控行为对所有环境一致，不存在那种不对称。
        """
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        lo = self.cfg.commands.parkour_min_vx
        # hi 兜底：play.py 的 CONTROL_ROBOT 分支会把 lin_vel_x 范围压成 [0,0]，
        # 那样 torch_rand_float(0.3, 0.0) 会反向取到 [0, 0.3]，指令悄悄失真。
        hi = max(self.command_ranges["lin_vel_x"][1], lo)
        self.commands[env_ids, 0] = torch_rand_float(lo, hi, (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        self.commands[env_ids, 1] = 0.
        self.commands[env_ids, 2] = 0.

    # ---------------- spawn / reset ----------------
    def _reset_root_states(self, env_ids):
        """出生在 goals[0]（跑道起点）而不是块中心——EP 的机器人也是从通道口出发。"""
        super()._reset_root_states(env_ids)
        if not self.custom_origins or len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        g0 = self.terrain_goals[self.terrain_levels[env_ids], self.terrain_types[env_ids], 0]
        jit = self.cfg.terrain.parkour_spawn_jitter
        self.root_states[env_ids, 0] = g0[:, 0] + torch_rand_float(
            -jit, jit, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 1] = g0[:, 1] + torch_rand_float(
            -jit, jit, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 2] = g0[:, 2] + self.base_init_state[2] + 0.05
        ids32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        # 从 1 开始：goals[0] 就是出生点本身，从 0 开始会在出生瞬间白送一次"到达"，
        # reached_goals 恒定虚高 1。第一个真正的目标是 goals[1]（第一级台阶）。
        self.cur_goal_idx[env_ids] = 1
        self.reached_goals[env_ids] = 0

    # ---------------- curriculum: 按走到第几个 goal 升降级 ----------------
    def _update_terrain_curriculum(self, env_ids):
        """EP 的课程是"走完这条通道就升级"。这里直接用 goal 计数，比投影距离更贴切：
        它天然只认"沿通道穿越"，横向乱走不会推进 goal 指针。"""
        if not self.init_done or len(env_ids) == 0:
            return
        if not hasattr(self, 'cur_goal_idx'):
            self._init_goal_buffers()
        reached = self.reached_goals[env_ids]
        move_up = reached >= self.cfg.terrain.parkour_goals_to_level_up
        move_down = (reached <= self.cfg.terrain.parkour_goals_to_level_down) & ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                         self.terrain_types[env_ids]]

    # ---------------- observations: 加入到 goal 的 delta yaw ----------------
    def compute_observations(self):
        if not hasattr(self, 'goal_delta_yaw'):
            self._init_goal_buffers()
            self._update_goals()
        # EP 观测的是"到 goal 的相对朝向"。用 cos/sin 而不是裸角度，避免 ±pi 处的跳变。
        goal_obs = torch.stack((torch.cos(self.goal_delta_yaw),
                                torch.sin(self.goal_delta_yaw)), dim=1)

        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            goal_obs,
        ), dim=-1)

        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights,
            -1, 1.) * self.obs_scales.height_measurements
        obs_buf = torch.cat((obs_buf, heights), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            goal_obs,
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

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()

        if self.cfg.env.frame_stack is not None:
            self.obs_history.append(obs_now)
            obs_buf_all = torch.stack([self.obs_history[i]
                                       for i in range(self.obs_history.maxlen)], dim=1)
            self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        else:
            self.obs_buf = obs_now

    # ---------------- rewards: EP 的两项核心 ----------------
    def _reward_tracking_goal_vel(self):
        """min(<d_hat, v_world>, vx_cmd) / vx_cmd，逐字对应 EP 的 _reward_tracking_goal_vel。

        与我们之前的 world_progress 的关键差别：d_hat 由【位置差】算出，机器人一旦
        横向偏离，d_hat 就转过来指回 goal，投影随之下降——这是"锚在方向"拿不到的
        反偏移信号。归一化到 [.., 1] 使慢速指令下完美跟随同样得满分。
        """
        if not hasattr(self, 'target_pos_rel'):
            return torch.zeros(self.num_envs, device=self.device)
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        d_hat = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        cmd = self.commands[:, 0]
        rew = torch.minimum(torch.sum(d_hat * cur_vel, dim=-1), cmd) / (cmd + 1e-5)
        return torch.clamp(rew, min=-1.0, max=1.0)

    def _reward_tracking_yaw(self):
        """exp(-|Δψ|)，Δψ 是到当前 goal 的方位角误差（EP 的 _reward_tracking_yaw）。"""
        if not hasattr(self, 'goal_delta_yaw'):
            return torch.zeros(self.num_envs, device=self.device)
        return torch.exp(-torch.abs(self.goal_delta_yaw))

    def _reward_goal_reached(self):
        """踩到一个新 goal 的稀疏奖励。EP 没有这一项（它靠 goal 推进本身改变 d_hat），
        这里保留一个很小的权重作为课程信号，可在 config 里置 0 完全对齐 EP。"""
        if not hasattr(self, 'cur_goal_idx'):
            return torch.zeros(self.num_envs, device=self.device)
        return (self.cur_goal_idx > 0).float() * 0.  # 由 config scale 控制，默认关闭

    # ---------------- debug 可视化：把 goal 画出来 ----------------
    def _draw_debug_vis(self):
        """在基类的高度图散点之外，额外画出本块地形的 goal 路点。

        Parkour 的行为完全由"当前 goal 在哪"决定，看不到 goal 就没法判断机器人是
        在朝目标走还是在乱走——这是 play 里最需要的一条信息。
        绿色小球 = 还没到的 goal，红色大球 = 当前正在追的那个。
        """
        super()._draw_debug_vis()
        if self.viewer is None or not hasattr(self, 'cur_goal_idx'):
            return
        goals = self._env_goals()          # (N, num_goals, 3)
        pending = gymutil.WireframeSphereGeometry(0.08, 6, 6, None, color=(0, 1, 0))
        current = gymutil.WireframeSphereGeometry(0.16, 8, 8, None, color=(1, 0, 0))
        for i in range(self.num_envs):
            gi = int(self.cur_goal_idx[i])
            g = goals[i].cpu().numpy()
            for k in range(g.shape[0]):
                geom = current if k == gi else pending
                pose = gymapi.Transform(gymapi.Vec3(g[k, 0], g[k, 1], g[k, 2] + 0.05), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[i], pose)

    # ---------------- 步态：堵住"单腿拖行"这个局部最优 ----------------
    def _reward_feet_air_time(self):
        """每只脚独立计腾空时长后相加，不再对两脚取 min（方案 A）。

        基类版本：
            in_mode_time = where(in_contact, feet_contact_time, feet_air_time)
            rew = min(where(single_stance, in_mode_time, 0), dim=1)
        它对两脚的 in-mode 时间取 min，而拖地脚【永远在接触】，它的 in_mode_time 是
        一路增长的 contact_time，min 于是总是取到摆动脚的 air_time —— "一只脚永远
        贴地 + 另一只脚迈步"因此拿到和真正交替迈步一样的满分。实测 model_999：
        左脚触地 80.2%、右脚 22.5%，不对称度 0.578，而该项照样给到 0.06~0.07。

        改法：只认【每只脚自己最近一次完整腾空的时长】，两脚各自封顶后相加再乘 0.5，
        使上限与基类的 0.5 一致。拖地脚从不腾空、last_air_time 恒为 0，贡献为零，
        整项直接少一半；真正交替的步态两脚都有腾空，不受影响。

        注意不能简单把 min 换成 sum：那样拖地脚的 contact_time 会被当作"in-mode
        时间"越滚越大，反而把拖行奖励成最优解。
        """
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        if not hasattr(self, 'last_air_time'):
            self.last_air_time = torch.zeros_like(self.feet_air_time)

        # 落地【边沿】检测必须在 += self.dt 之前，与上游 legged_gym 的 first_contact
        # 同序。放到 += 之后就不是边沿了：feet_air_time 在触地步末尾被清零，下一步
        # 加上 dt 后又 >0，于是整个支撑相每一步都判成"刚落地"，把 last_air_time 反复
        # 覆盖成 dt(0.02s)，而真实每步腾空是 0.10~0.13s —— 该项因此只输出上限的 2.4%。
        touchdown = contact_filt & (self.feet_air_time > 0)
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.last_air_time = torch.where(touchdown, self.feet_air_time, self.last_air_time)
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt

        # 时效性：last_air_time 只在落地那一刻更新，不清理的话一只脚哪怕十秒没抬过
        # 也还在吃十秒前那次的信用——拖地腿因此照拿满分，单腿步态治不好。连续触地
        # 超过 stale_time 即判定不再迈步、信用清零；正常支撑相远短于该阈值。
        stale = self.feet_contact_time > self.cfg.rewards.feet_air_time_stale
        self.last_air_time = torch.where(stale, torch.zeros_like(self.last_air_time),
                                         self.last_air_time)

        single_stance = torch.sum(contact_filt.int(), dim=1) == 1
        per_foot = torch.clamp(self.last_air_time, max=0.5)
        rew = 0.5 * torch.sum(per_foot, dim=1) * single_stance
        return rew
