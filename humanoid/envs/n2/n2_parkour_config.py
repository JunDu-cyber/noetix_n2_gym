from humanoid.envs.n2.n2_perceptive_config import N2PerceptiveCfg, N2PerceptiveCfgPPO


class N2ParkourCfg(N2PerceptiveCfg):
    """Extreme Parkour(arXiv:2309.14341)架构的配置。

    独立于 n2_perceptive 注册成新任务，不动原有的感知任务——后者已经有一份能训到
    terrain_level 5.49 的成果，不该被这次架构实验波及。

    与 n2_perceptive 的三处根本差异（都是 EP 的要件）：
      指令：只有前进速度，lin_vel_y 和 ang_vel_yaw 都是 [0,0]
      目标：goal 路点位置，而不是速度指令方向
      地形：中央通道 + y 向 pad + goal 落在障碍上，横向绕行被几何堵死
    """

    class env(N2PerceptiveCfg.env):
        # 39 本体 + 2 到goal的delta-yaw(cos/sin) + 96 高度 = 137
        frame_stack = 10
        num_single_obs = 39 + 2 + 96
        num_observations = int(frame_stack * num_single_obs)
        # 特权观测同步 +2
        num_privileged_obs = N2PerceptiveCfg.env.num_privileged_obs + 2

    class terrain(N2PerceptiveCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        max_init_terrain_level = 0

        # EP 的块是 18m。这里 8m：起步平台 2.5m + 8 级台阶(每级 0.2~0.4m，共约 2.4m)
        # + 末端平台，8m 绰绰有余，且地形总面积比 12m 小三分之一、显存更宽松。
        terrain_length = 8.
        terrain_width = 4.
        num_rows = 10
        num_cols = 10

        # ParkourTerrain 恒定生成 parkour_step_terrain，不再按比例分派多种地形，
        # 所以 terrain_proportions 在这个任务里不起作用（保留以兼容基类构造）。
        terrain_proportions = [0., 0., 0., 0., 0., 0., 0., 1.0, 0.]

        # ---- 以下取自 EP 源码 parkour_step_terrain 的默认值本身 ----
        num_goals = 10                      # 起步平台 + 8 级台阶 + 末端平台 = EP 的 num_stones=8
        parkour_platform_len = 2.5          # EP: platform_len=2.5
        parkour_x_range = (0.2, 0.4)        # EP: x_range=[0.2,0.4]，每级台阶的踏面长度
        parkour_y_range = (-0.15, 0.15)     # EP: y_range=[-0.15,0.15]
        parkour_pad_width = 0.1             # EP: pad_width=0.1，最外缘细边框
        parkour_pad_height = 0.5            # EP: pad_height=0.5
        # 台阶高度随难度从 5cm 线性升到 EP 的 20cm（唯一的课程维度）
        parkour_step_height_range = [0.05, 0.20]

        # ---- 唯一因机器人形态而偏离 EP 的一项 ----
        # EP 的 half_valid_width=[0.45,0.5] ⇒ 通道宽 0.9~1.0m，是给 A1 四足（体宽约
        # 0.3m）设计的。N2 是人形，双脚横向间距加上摆动余量远超四足，0.9m 通道会让
        # 它频繁踩空到通道外的低地。这里放宽到 [0.7,0.8] ⇒ 通道宽 1.4~1.6m，
        # 与 Robot Parkour Learning 给四足用的 1.6m 单行道同量级，按人形比例是合适的。
        # 若发现机器人仍在通道外行走，再收窄回 EP 的原值。
        parkour_half_valid_width = (0.7, 0.8)
        # 出生点抖动(m)
        parkour_spawn_jitter = 0.3
        # 课程：走到第几个 goal 才升级 / 不足几个就降级
        parkour_goals_to_level_up = 5
        parkour_goals_to_level_down = 1

    class commands(N2PerceptiveCfg.commands):

        stairs_forward_only = False
        # 站立指令比例设 0：EP 不发站立指令，机器人始终在通道里前进
        standing_prob = 0.02
        zero_vx_prob = 0.0
        zero_wz_prob = 0.0
        # 前进速度下限(m/s)
        parkour_min_vx = 0.3

        class ranges(N2PerceptiveCfg.commands.ranges):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]      # EP: lin_vel_y=[0,0]
            ang_vel_yaw = [0.0, 0.0]    # EP: ang_vel_yaw=[0,0]

    class rewards(N2PerceptiveCfg.rewards):
        # goal 到达半径(m)：进到这个范围内就切换到下一个 goal
        goal_reach_dist = 0.4

        class scales(N2PerceptiveCfg.rewards.scales):
            # ---- EP 的两项核心，权重取自 EP 自己的配置 ----
            tracking_goal_vel = 1.5     # EP: tracking_goal_vel = 1.5
            tracking_yaw = 0.5          # EP: tracking_yaw = 0.5
            goal_reached = 0.0          # 非 EP 项，默认关闭

            # ---- 关掉 n2_perceptive 那套世界系反绕路奖励 ----
            # 它们解决的是"全向指令 + 定向地形"的错配，而 Parkour 架构从源头
            # 消除了这个错配（指令只前进、goal 锚定位置），再叠加只会互相干扰。
            world_progress = 0.0
            world_heading = 0.0
            anti_freeze = 0.0

            # ---- base 系速度跟踪降权 ----
            # EP 保留但次要；主目标已经由 tracking_goal_vel 承担
            tracking_lin_vel = 0.5
            tracking_ang_vel = 0.3


class N2ParkourCfgPPO(N2PerceptiveCfgPPO):
    class runner(N2PerceptiveCfgPPO.runner):
        experiment_name = 'n2_parkour'
        empirical_normalization = True
