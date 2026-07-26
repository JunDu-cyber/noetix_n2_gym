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
        # goal 到达半径(m)。必须【小于最小 goal 间距】——间距由 terrain.parkour_x_range
        # 决定，下限 0.2m，实测最小 0.22m。原值 0.5 让 9 对相邻 goal 里 7 对(78%)落在
        # 半径内，指针连跳、课程虚高：训练报 terrain_level 6.3（14.5cm 台阶），而同期
        # checkpoint 实测在 8.8cm 台阶只有 11% 存活、16.3cm 时 0% 越过台阶起点。
        # 详见 N2ParkourEnv._update_goals；环境构造时有 assert 兜底，改
        # parkour_x_range 时会强制复核这个耦合。
        goal_reach_dist = 0.15
        # "越过即算到达"的横向门限(m)。半径缩小后机器人可能擦过 goal 而进不了圈，
        # goal 留在身后会让 target_pos_rel 掉头、要求它转身回去；"越过"分支避免这点。
        # 该门限防止绕到通道外(低地)沿 y 平移把 goal 一路"越过"。通道半宽 0.7~0.8m。
        goal_pass_lateral_tol = 0.5
        # _reward_feet_air_time：单只脚连续触地超过这么久(s)就判定"已不再迈步"，
        # 把它上一次的腾空信用清零。正常支撑相约 0.3~0.6s，取 1.0s 留余量。
        feet_air_time_stale = 1.0

        class scales(N2PerceptiveCfg.rewards.scales):
            # ---- EP 的两项核心，权重取自 EP 自己的配置 ----
            # 1.5 -> 4.0。实测 model_600（速度 0.228 m/s、净进展仅 0.08~0.12 m/s）的
            # 每步奖励分解：正项合计 0.0861，其中【站着不动就能拿的】占 0.0701（81%）——
            # default_joint_pos 0.0194 + feet_contact 0.0189 + orientation 0.0174 +
            # tracking_yaw 0.0093 + tracking_ang_vel 0.0051。而 tracking_goal_vel 只有
            # 0.0103，因为它的 raw 只拿到 0.039/1.0。
            # 最刺眼的一对：tracking_yaw 站着朝向 goal 就拿 95%，tracking_goal_vel 只有 4%，
            # 两者最终贡献几乎相同。策略于是收敛到"站稳、朝向 goal、极慢挪动"——正是
            # arXiv:2010.04304 描述的 standing-still 局部最优（"balances but never steps
            # forward"）。iter 640 时 ep_len 已达上限的 93%、noise_std 降到 0.709，
            # 说明它对这套不动的策略非常自信，继续训不会自行好转。
            # 4.0 使满速时该项 0.08/步 = 姿态项的 114%，让"朝 goal 前进"成为主导目标；
            # 该项有界 [-1,1]，最坏单步 -0.08，不存在尖峰通道。
            tracking_goal_vel = 4.0
            tracking_yaw = 0.5          # EP: tracking_yaw = 0.5
            goal_reached = 0.0          # 非 EP 项，默认关闭

            # ---- 关掉 n2_perceptive 那套世界系反绕路奖励 ----
            # 它们解决的是"全向指令 + 定向地形"的错配，而 Parkour 架构从源头
            # 消除了这个错配（指令只前进、goal 锚定位置），再叠加只会互相干扰。
            world_progress = 0.0
            world_heading = 0.0
            anti_freeze = 0.0

            # ---- 步态：堵住单腿拖行 ----
            # contact_no_vel 罚的是"带速度的接触"，也就是拖蹭。它一直在罚，只是
            # 太小：实测坏掉的 model_999 上只有 -0.004/步，占正项合计的 5.1%，
            # 完全压不住。x5 提到 -10 后约占 26%，是能起作用又不至于把总奖励压到
            # only_positive_rewards 截断线以下的量级（x10 就到 51%，太激进，
            # perceptive 上吃过这个亏）。
            # 要说明的是它【不区分左右】：实测拖地脚占该项 53%、摆动脚 47%，
            # 因为摆动脚在触地/蹬离瞬间同样有带速度的接触。所以这一项不是
            # 对称性药方，而是把"拖蹭"整体变贵、推动两脚都干净落地。
            # 真正打击不对称的是上面改过的 feet_air_time（拖地脚贡献归零）。
            # -6 -> -2（改回基类值）。当初提到 -6 是为压制拖蹭，但实测在当前均衡点上
            # 它只有 -0.0027/步（正项的 3%），已经不是约束；而机器人一旦真开始走快，
            # 触地速度上升会让它重新变成阻力——留着就是给提速埋雷。
            # 对称性已由 feet_air_time 的改动（方案A）保住，不依赖这一项。
            contact_no_vel = -2

            # ---- base 系速度跟踪降权 ----
            # EP 保留但次要；主目标已经由 tracking_goal_vel 承担
            tracking_lin_vel = 0.5
            tracking_ang_vel = 0.3


class N2ParkourCfgPPO(N2PerceptiveCfgPPO):
    class runner(N2PerceptiveCfgPPO.runner):
        experiment_name = 'n2_parkour'
        empirical_normalization = True

    class policy(N2PerceptiveCfgPPO.policy):
        # 主干 [256,128] -> [512,256,128]，与 Extreme Parkour / legged_gym 上游一致。
        # 我们的输入维度(1370)本来就比 EP 的还大，却用了更小的网络，容量与输入规模
        # 明显不匹配。
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

        # 高度图编码器：把 frame_stack x 96 = 960 维高度图先压到 32 维再与本体感知
        # 拼接（EP 的 scan_encoder 是 [128,64,32]，把 132 维 scandots 压到 32）。
        # 不启用时 actor 第一层是 1370x256=350720，占 actor 参数的 91%；启用后主干
        # 输入降到 41x10+32=442。
        scan_encoder_dims = [128, 64, 32]
        # 观测布局（必须与 N2ParkourEnv.compute_observations 一致）：
        # 每帧 = [cmd3+angvel3+grav3+dofpos10+dofvel10+act10+goal2 = 41] + [高度96]
        frame_stack = 10
        n_proprio = 41
        n_scan = 96
