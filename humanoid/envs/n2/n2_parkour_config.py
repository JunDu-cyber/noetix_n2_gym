from humanoid.envs.n2.n2_perceptive_config import N2PerceptiveCfg, N2PerceptiveCfgPPO


class N2ParkourCfg(N2PerceptiveCfg):
    """Extreme Parkour(arXiv:2309.14341)架构的配置。
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

        # 起步平台 2.5m + 8 级台阶(约 2.4m) + 末端平台，8m 足够(EP 用 18m)。
        terrain_length = 8.
        terrain_width = 4.
        num_rows = 10
        num_cols = 10

        # ParkourTerrain 恒定生成 parkour_step_terrain
        terrain_proportions = [0., 0., 0., 0., 0., 0., 0., 1.0, 0.]

        # ---- 以下取自 EP 源码 parkour_step_terrain 的默认值本身 ----
        num_goals = 10                      # 起步平台 + 8 级台阶 + 末端平台 = EP 的 num_stones=8
        parkour_platform_len = 2.5          # EP: platform_len=2.5
        parkour_x_range = (0.2, 0.4)        # EP: x_range=[0.2,0.4]，每级台阶的踏面长度
        parkour_y_range = (-0.15, 0.15)     # EP: y_range=[-0.15,0.15]
        parkour_pad_width = 0.1             # EP: pad_width=0.1，最外缘细边框
        parkour_pad_height = 0.5            # EP: pad_height=0.5

        parkour_step_height_range = [0.05, 0.20]

        parkour_half_valid_width = (0.7, 0.8)
        # 出生点抖动(m)
        parkour_spawn_jitter = 0.3
        # 课程
        parkour_goals_to_level_up = 5
        parkour_goals_to_level_down = 1

    class commands(N2PerceptiveCfg.commands):
        # 本任务整体只发前进指令，不需要按地形列区分。
        stairs_forward_only = False
        # EP 完全不发站立指令；这里留一点点，其余时间在通道里前进。
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
        # 【必须小于最小 goal 间距】
        goal_reach_dist = 0.15
        # 越过即算到达
        goal_pass_lateral_tol = 0.5

        feet_air_time_stale = 1.0

        class scales(N2PerceptiveCfg.rewards.scales):

            tracking_goal_vel = 3.5
            tracking_yaw = 0.5          # EP: tracking_yaw = 0.5
            goal_reached = 0.0          # 非 EP 项，默认关闭


            world_progress = 0.0
            world_heading = 0.0
            anti_freeze = 0.0


            contact_no_vel = -2

            # base 系速度跟踪降权：主目标已由 tracking_goal_vel 承担。
            tracking_lin_vel = 0.5
            tracking_ang_vel = 0.3


class N2ParkourCfgPPO(N2PerceptiveCfgPPO):
    class runner(N2PerceptiveCfgPPO.runner):
        experiment_name = 'n2_parkour'
        empirical_normalization = True

    class policy(N2PerceptiveCfgPPO.policy):
        # 与 Extreme Parkour / legged_gym 上游一致。
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        # 把 frame_stack x 96 = 960 维高度图压到 32 维再与本体感知拼接(EP 的 scan_encoder)。
        # 不启用时 actor 第一层 1370x512 占参数 91%；启用后主干输入降到 41x10+32=442。
        scan_encoder_dims = [128, 64, 32]
        # 观测布局（必须与 N2ParkourEnv.compute_observations 一致）：
        # 每帧 = [cmd3+angvel3+grav3+dofpos10+dofvel10+act10+goal2 = 41] + [高度96]
        frame_stack = 10
        # 观测布局必须与 N2ParkourEnv.compute_observations 一致：
        # 每帧 = [cmd3+angvel3+grav3+dofpos10+dofvel10+act10+goal2 = 41] + [高度96]。
        # 构造时 assert frame_stack*(n_proprio+n_scan) == num_actor_obs。
        n_proprio = 41
        n_scan = 96
