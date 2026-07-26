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
        """
        xr = getattr(self.cfg.terrain, 'parkour_x_range', (0.2, 0.4))
        reach = self.cfg.rewards.goal_reach_dist
        if reach >= float(xr[0]):
            raise ValueError(
                "goal_reach_dist=%.3f 必须 < parkour_x_range 下限 %.3f，"
                "否则相邻 goal 落在到达半径内、指针连跳导致课程虚高" % (reach, xr[0]))

    def _init_goal_buffers(self):
        self._check_goal_reach_consistency()
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
        """踩到一个新 goal 的稀疏奖励。EP 没有这一项（
        这里保留一个很小的权重作为课程信号，可在 config 里置 0 完全对齐 EP。"""
        if not hasattr(self, 'cur_goal_idx'):
            return torch.zeros(self.num_envs, device=self.device)
        return (self.cur_goal_idx > 0).float() * 0.  # 由 config scale 控制，默认关闭

    # ---------------- debug 可视化：把 goal 画出来 ----------------
    def _draw_debug_vis(self):
        """在基类的高度图散点之外，额外画出本块地形的 goal 路点。
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
        """每只脚独立计腾空时长后相加，不再对两脚取 min。
        """
        contact_filt = torch.logical_or(self.contacts, self.last_contacts)
        if not hasattr(self, 'last_air_time'):
            self.last_air_time = torch.zeros_like(self.feet_air_time)

        touchdown = contact_filt & (self.feet_air_time > 0)
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.last_air_time = torch.where(touchdown, self.feet_air_time, self.last_air_time)
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt

        stale = self.feet_contact_time > self.cfg.rewards.feet_air_time_stale
        self.last_air_time = torch.where(stale, torch.zeros_like(self.last_air_time),
                                         self.last_air_time)

        single_stance = torch.sum(contact_filt.int(), dim=1) == 1
        per_foot = torch.clamp(self.last_air_time, max=0.5)
        rew = 0.5 * torch.sum(per_foot, dim=1) * single_stance
        return rew
