from humanoid.envs.n2.n2_10dof_config import N2_10dof_Cfg, N2_10dof_CfgPPO

class N2PerceptiveCfg(N2_10dof_Cfg):
    class env(N2_10dof_Cfg.env):
        # 96 = len(measured_points_x)=12 × len(measured_points_y)=8

        frame_stack = 10                                        # 开启帧堆叠(改这个值即可)
        num_single_obs = 39 + 96                              # 135:本体感知39 + 高度96(每帧都含高度)
        num_observations = int(frame_stack * num_single_obs)  # 675 = 5 × 135


    class terrain(N2_10dof_Cfg.terrain):
        measure_heights = True              # blind 里本来也是 True,显式写上保险
        debug_viz = True
        curriculum = True

        # 初始地形等级
        max_init_terrain_level = 0 #10
        # 地形比例分布 [平面; 障碍物; 均匀; 上坡; 下坡, 上楼梯, 下楼梯]
        # terrain_proportions = [0.7, 0.0, 0.2, 0.1, 0.0, 0., 0.]
        terrain_proportions = [0., 0.0, 0.1, 0.0, 0.0, 0.05, 0.15, 0.25, 0.15]

    class noise(N2_10dof_Cfg.noise):
        class noise_scales(N2_10dof_Cfg.noise.noise_scales):
            height_measurements = 0.0       # privileged = 干净真值,必须为0

    class rewards(N2_10dof_Cfg.rewards):
        # ε : if terrain under a sole sample sits more than this (m) below the
        # foot, the sample is "over a void" → improper placement. (paper's ε)
        foothold_depth_tol = 0.04
        # foot sole footprint used to lay out the n sample points (metres).
        # N2 "ankle" foot ~0.20 x 0.10; tune to your collision mesh.
        foot_length = 0.20
        foot_width = 0.10
        foot_n_x = 3  # samples along length
        foot_n_y = 2  # samples along width  → n = 6 per foot

        # 参考朝向 yaw_ref 相对实际 yaw 的泄漏钳制上限（rad），见
        # N2PerceptiveEnv._update_world_reference。取 π/2 而不是更小的值：
        # 在 π/2 处，绕路转身 90° 仍然把 world_progress 打到 ~0（cos(π/2)）、
        # world_heading 打到 ~0.007，反绕路信号完整保留；只有超过 90° 之后
        # 惩罚才饱和，而那时惩罚已经是最大的了。钳制的作用只是防止机器人
        # 物理上跟不上偏航指令（摔倒、楼梯上难转身）时 yaw_ref 以 1 rad/s
        # 跑飞 15 秒，把不可控的方差灌进回报。
        world_heading_max_err = 1.57
        # _reward_stand_still 的两个参数，见 N2PerceptiveEnv._reward_stand_still。
        # 该项 = sum|dof_pos-default| + w*sum(dof_vel²)，速度项二次且无上界。
        # 在训练真实条件下（采样动作，带策略探索噪声——这是 dof_vel 的主要
        # 来源，用确定性推理去测会完全漏掉）实测 0723_19-51-09_ 的 checkpoint：
        # 站立环境 raw 均值 88.8 / 中位数 43.5 / p95 338，每步总奖励在截断前
        # 均值 -0.162，**75.3% 的步被 only_positive_rewards 削成 0**（运动
        # 环境只有 17.0%）。站立状态因此长期没有梯度，姿态从未被真正训练，
        # 表现为部署时"站着抖、一给速度指令就不抖"。
        # 光封顶没用（封在 80 仍有 75% 被截断，问题出在中位数不在尾部），
        # 必须给速度项降权把站立状态拉回正区间；降权也优于直接调小 scale，
        # 因为硬封顶以上 dof_vel 的梯度恰好为 0，反而丢掉了压制抖动的信号。
        stand_still_vel_weight = 0.05
        stand_still_max = 20.0

        class scales(N2_10dof_Cfg.rewards.scales):
            foothold = -0.15  # sign lives here; reward fn returns +count

            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.0

            # 反"绕路/后退"：用按指令偏航率积分的参考朝向 yaw_ref 构造世界系
            # 目标方向，奖励世界系实际速度/朝向对它的跟踪（仿 Extreme Parkour,
            # arXiv:2309.14341 的 world-frame progress reward）。完全遵循三路
            # 指令的机器人两项都拿满分，只有"未被指令的"偏航偏移（绕路转身）
            # 或横移/后退才掉分。详见 N2PerceptiveEnv._update_world_reference。
            #
            # 标定值必须随 round-4 重构一起下调。旧的 5.0/2.5 是在**冻结**目标
            # 那版上调出来的，而那版的信号实测封顶在 0.517/0.408（蒙特卡洛算出
            # 的"完美遵循指令、绝不绕路"的机器人得分，实际策略拿到 0.434/0.388，
            # 即该项当时只是被偏航指令饱和、对绕路零信息量）。重构后同样的 5.0/
            # 2.5 能达到 ~2.75/2.5，是原来的 5~6 倍奖励质量，会盖过整个 stack
            # （tracking_lin_vel 1.4 + tracking_ang_vel 1.6 满打满算才 3.0）。
            # 1.5/1.0 让两项的可达幅值与 tracking 系列同量级，也与 Extreme
            # Parkour 自己的 tracking_goal_vel=1.5 / tracking_yaw=0.5 一致。
            # 判据：跑 1500~2000 iter 后这两项若明显**超过** 0.517/0.408，说明
            # 奖励终于开始度量绕路了；同时 noise_std 应止涨（旧版 1.0→2.61 单调
            # 上升）。
            world_progress = 1.8
            world_heading = 1.4

            # 障碍物/楼梯通行相关：碰撞与踢竖面惩罚（原本已实现但未启用）
            # collision: 参考 legged_gym 上游 base 默认值及 anymal_c/a1 rough
            # terrain 配置（均未覆盖此值，直接沿用 -1.0 用于实际训练）
            collision = -1.0
            # stumble: 对应 _reward_stumble（注意 key 必须是 stumble，不是
            # legged_gym 里名字对不上的 feet_stumble，否则会 AttributeError）。
            # 上游没有任何参考配置启用过这一项，这里的数值是按 collision 同量级
            # 给的经验起点，需要在下一轮训练里看 TensorBoard 再调
            stumble = -2.5


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'
        # 高度图(96维,占每帧135维的71%)缩放后典型幅度比其余本体感知维度大很多,
        # 基类默认不做观测归一化(Identity),这里为 perceptive 单独打开经验归一化
        empirical_normalization = True