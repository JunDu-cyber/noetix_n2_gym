from humanoid.envs.n2.n2_10dof_config import N2_10dof_Cfg, N2_10dof_CfgPPO

class N2PerceptiveCfg(N2_10dof_Cfg):
    class env(N2_10dof_Cfg.env):
        # 每帧 = 本体感知 39 + 高度图 96(= measured_points_x 12 × measured_points_y 8)
        frame_stack = 10
        num_single_obs = 39 + 96
        num_observations = int(frame_stack * num_single_obs)

    class terrain(N2_10dof_Cfg.terrain):
        measure_heights = True              # blind 里本来也是 True,显式写上保险
        debug_viz = True
        curriculum = True

        # 初始地形等级
        max_init_terrain_level = 0 #10
        # [平地;离散障碍;均匀;上坡;下坡;金字塔上/下楼梯;直行上/下楼梯]，累加须为 1.0。
        # 不足 1.0 的部分会落进 make_terrain 的 else:pass 变成纯平地，既浪费环境又把
        # terrain_level 均值拉高(平地列轻松满级、掩盖楼梯列卡在低级)。
        terrain_proportions = [0.0, 0.15, 0.0, 0.0, 0.0, 0.05, 0.15, 0.40, 0.25]

        # 课程升级所需的方向性进展(m)。基类是 env_length/2，配合中心出生对早期楼梯
        # 过于苛刻，会把楼梯列钉死在 0 级。
        curriculum_up_distance = 1.6

        # 8m 块，与 legged_gym / IsaacLab 对齐。4m 块下 directional_stairs 扣掉平台只剩
        # 2.5m 可爬段，课程没有上推空间。只覆盖 perceptive，盲策略仍是 4m。
        # 代价：地形面积变 4 倍，显存吃紧时调小 num_rows/num_cols。
        terrain_length = 8.
        terrain_width = 8.

        # directional_stairs 的 -x 端底部平台，机器人在此出生。
        # 【必须有】：isaacgym 的 stairs_terrain 是地形族里唯一不接受 platform_size 的
        # 函数，而出生高度沿用 legged_gym 的 env_origin_z = max(中心±1m 窗口)，该式默认
        # 中心是平的；单调楼梯上它会取到前方最高台阶，实测出生悬空 0.2~0.8m(台阶越高
        # 越严重)，等于每次 reset 都自由落体。
        # 保持 1.5 不随块尺寸放大：出生点在平台中心，plat/2=0.75m 是到第一级台阶的
        # 距离，curriculum_up_distance 的语义依赖于此。
        stairs_platform_size = 1.5

    class commands(N2_10dof_Cfg.commands):
        # 站立指令比例。基类 0.20 等于拿 20% 算力专训"站着不动"，而感知爬楼的失败模式
        # 恰恰是"一感知到楼梯就退回站立"。IsaacLab 官方人形粗糙地形基线用 0.02。
        standing_prob = 0.05

        # 只在直行楼梯列(index 7/8)把指令锁成纯前进(vy=0, wz=0)，让"定向地形"配
        # "定向指令"(Extreme Parkour 的组合)，而不是基类的全向指令。其余列不受影响。
        #
        # 【默认关闭，开启前必读】它会踩到一个隐性耦合：N2_10dof_Env.
        # _reward_default_joint_pos（权重 1.0，第三大正项）的门控是
        #     rew[any(|commands[:,[1,2]]| > min_cmd_vel) or standing_cmd] = 1.0
        # ——有横移/偏航/站立指令就白送满分，只有纯前进才真考核髋关节偏差。本开关
        # 恰好把 vy/wz 精确置零，实测白送比例 92% -> 6%。而每步净奖励本就贴着
        # only_positive_rewards 的截断线，抽掉这块正项后总和被钳成 0、梯度消失、
        # noise_std 单调发散。已烧掉两轮训练。
        # 要开启必须先重写 _reward_default_joint_pos 让纯前进指令也享有豁免。
        stairs_forward_only = False
        # 楼梯列前进速度重采样下限(m/s)。不能取 0：低于 min_cmd_vel=0.2 会被判成站立，
        # 把楼梯列站立比例抬到近 30%，抵消 standing_prob 的下调。
        stairs_min_vx = 0.25

    class domain_rand(N2_10dof_Cfg.domain_rand):
        # 关掉每次 reset 重抽摩擦：_refresh_actor_rigid_shape_props 的 per-env Python 循环
        # 在早期 episode 很短时会吃掉大部分采样时间(含 reset 的步 296ms vs 52ms)。
        # 建环境时的 per-env 随机化仍生效，策略看到的摩擦分布不变。
        refresh_shape_props_on_reset = False

    class noise(N2_10dof_Cfg.noise):
        class noise_scales(N2_10dof_Cfg.noise.noise_scales):
            height_measurements = 0.0       # privileged = 干净真值,必须为0

    class rewards(N2_10dof_Cfg.rewards):
        # 旧离散实现的阈值，已被 foothold_flat_k 的平滑形式取代，保留仅作量纲参考。
        foothold_depth_tol = 0.04
        # _reward_foothold 的平滑系数：raw = 1 - exp(-k * 平均悬空深度)。
        # k=20 使旧阈值 4cm 落在 0.55 附近(动态范围中段)。
        foothold_flat_k = 20.0
        # foot sole footprint used to lay out the n sample points (metres).
        # N2 "ankle" foot ~0.20 x 0.10; tune to your collision mesh.
        foot_length = 0.20
        foot_width = 0.10
        foot_n_x = 3  # samples along length
        foot_n_y = 2  # samples along width  → n = 6 per foot

        # yaw_ref 相对实际 yaw 的泄漏钳制上限(rad)。取 π/2：在该处绕路转身 90° 仍把
        # world_progress 打到 ~0、world_heading 打到 ~0.007，反绕路信号完整保留；
        # 钳制只为防止机器人物理上跟不上偏航指令时 yaw_ref 跑飞、灌入不可控方差。
        world_heading_max_err = 1.57
        # _reward_stand_still = sum|dof_pos-default| + w*sum(dof_vel²)，速度项二次且无上界。
        # 原权重下站立环境 75% 的步被 only_positive_rewards 削成 0，站立姿态从未被真正
        # 训练。硬封顶无效(问题在中位数不在尾部)，必须降权。
        stand_still_vel_weight = 0.05
        stand_still_max = 20.0


        # _reward_anti_freeze 的饱和阈值(m/s)：命令方向的世界系前进速度达到即拿满分。
        # 目的是打破"看到楼梯就原地站死"的局部最优——only_positive_rewards 把站立总回报
        # 钳到 0，惩罚被一并钳掉没有梯度；正奖励且低速饱和才能把"动"抬到"冻"之上。
        anti_freeze_speed = 0.15

        # 自重 33.2kg = 325N，阈值卡在自重之下意味着单脚支撑期光站着就扣分——这也是该项
        # 此前呈现"地形无关"的原因(它是恒定步态税，不是踩楼梯的代价)。
        max_contact_force = 400.
        # 每只脚超出部分的上限(N)。基类不封顶，摔倒砸地时单步惩罚能盖过整个正项栈。
        feet_contact_force_max_excess = 200.

        # _reward_world_progress 归一化分母的下限(m/s)。min_cmd_vel 只约束三维指令模长，
        # wz 主导的指令仍可能让 |v_xy| 接近 0，故给分母兜底。
        world_progress_min_speed = 0.1

        class scales(N2_10dof_Cfg.rewards.scales):

            foothold = -0.5

            feet_air_time = 4.0

            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.8


            world_progress = 0.8
            world_heading = 1.0

            # 与 tracking_lin_vel 同量级即可，太大会催生莽撞前冲。
            anti_freeze = 1.0

            collision = -1.0
            # key 必须是 stumble，不是 legged_gym 里名字对不上的 feet_stumble
            # (基类 scales 里叫 feet_stumble，但本仓库的方法名是 _reward_stumble)。
            stumble = -2.0


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'
        # 高度图占每帧 71% 且幅度远大于其余维度，基类默认 Identity 归一化不够用。
        empirical_normalization = True