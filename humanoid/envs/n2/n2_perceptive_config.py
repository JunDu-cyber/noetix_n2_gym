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
        # 地形比例分布 [平地; 离散障碍; 均匀; 上坡; 下坡; 金字塔上楼梯; 金字塔下楼梯;
        #              直行上楼梯; 直行下楼梯]。累加必须到 1.0：旧值只到 0.70，剩下
        #              30% 落进 make_terrain 的 else:pass = 纯平地，既浪费环境、又把
        #              terrain_level 均值拉高（平地列轻松升满级、掩盖楼梯列卡在低级）。
        #              现在用离散障碍(index 1，"大大小小方块高地")替掉那 30% 空平地，
        #              并把权重压到直行楼梯上——它是要攻克的目标地形。
        # [平地; 障碍物; 均匀; 上坡; 下坡, 上楼梯, 下楼梯]（旧 7 项注释保留供参考）
        # terrain_proportions = [0.7, 0.0, 0.2, 0.1, 0.0, 0., 0.]
        terrain_proportions = [0.0, 0.15, 0.0, 0.0, 0.0, 0.05, 0.15, 0.40, 0.25]

        # 课程升级所需的方向性进展阈值（m），见 N2PerceptiveEnv._update_terrain_curriculum。
        # 基类是 env_length/2=2m；配合中心出生，2m 意味着要爬完整个上半块楼梯才升级，
        # 对早期楼梯太苛刻，把楼梯列钉死在 0 级。降到 1.5m 让"爬了一段真台阶"就能升级，
        # 课程得以逐级把机器人推上更高楼梯。若仍卡住可再调低。
        curriculum_up_distance = 1.6

        # 地形块 4m -> 8m，与"亲兄弟"对齐：legged_gym 用 terrain_length/width = 8.，
        # IsaacLab 的 ROUGH_TERRAINS_CFG 也是 8m 块。4m 块下 directional_stairs 去掉
        # 1.5m 平台只剩 2.5m 可爬段，机器人爬几级就到顶；8m 块把可爬段拉到 6.5m
        # （踏面 0.3~0.45m ≈ 14~21 级），课程才有持续往上推的空间。
        # 只覆盖 perceptive，盲策略 n2_10dof/n2 仍是 4m，不受影响。
        # 代价：地形总面积变 4 倍（10x10 块 = 80x80m），trimesh 顶点数同步上升，
        # 建图时间与显存都会涨；若显存吃紧就调小 num_rows/num_cols。
        terrain_length = 8.
        terrain_width = 8.

        # directional_stairs 的 -x 端底部平台尺寸（m），机器人在这块平台上出生。
        # 【为什么必须有】isaacgym 的 stairs_terrain 是地形族里唯一不接受 platform_size
        # 的函数，而出生高度沿用 legged_gym 的 env_origin_z = max(中心±1m 窗口)，该式
        # 默认中心是平的；在单调楼梯上它取到前方 1m 处的最高台阶，实测出生悬空
        # 台阶5cm→0.20m / 10cm→0.40m / 20cm→0.80m，即每次 reset 都自由落体。
        # legged_gym 每个子地形都传 platform_size=3./4.、IsaacLab pyramid_stairs 用
        # platform_width=3.0，都是在维持"出生在平地上"这个契约。
        # 保持 1.5 而不随块尺寸放大到 3.0：出生点在平台中心，平台中心到楼梯起点的
        # 距离 = plat/2 = 0.75m，正好让 curriculum_up_distance=1.6 的语义不变
        # （1.6-0.75=0.85m 才是真正踩在台阶上的行进距离）。若放大到 3.0，光走完平台
        # 就要 1.5m，1.6 的阈值几乎不用爬台阶就能升级，课程会被架空。
        stairs_platform_size = 1.5

    class commands(N2_10dof_Cfg.commands):
        # 站立指令比例 0.20 -> 0.05。基类给了 20% 的环境"全部命令为 0"，感知爬楼
        # 任务里这等于拿 20% 的算力专门训练"站着不动"——而 MuJoCo 复现出来的失败
        # 模式恰恰是"一感知到楼梯就退回站立"（出生在楼梯上、给前进指令，vx≈0 站
        # 30 秒）。站立不是它不会走，是它被训得太会站了。
        # 参照物：IsaacLab 官方人形粗糙地形基线 Isaac-Velocity-Rough-H1-v0 用的是
        # rel_standing_envs=0.02（2%），比这里低一个数量级。取 0.05 作为折中，既
        # 大幅削掉这个吸引子的训练量，又保留一点站立能力（部署时仍需要能站住）。
        standing_prob = 0.05

        # ---- 路线 B：只在直行楼梯列(index 7/8)把指令锁到 +x ----
        # 起因：directional_stairs 沿 y 完全恒定，是【定向地形】，而基类发【全向随机
        # 指令】——这两者互斥。四个亲兄弟没有一个用这个组合：legged_gym / IsaacLab /
        # Oli 都是"对称地形(金字塔) + 全向指令"，Extreme Parkour / Robot Parkour 是
        # "定向地形 + 定向指令(lin_vel_y=[0,0], ang_vel_yaw=[0,0])"。我们落在了
        # 错配的那一格，症状就是楼梯列升级了却没学会爬、高台阶上转弯绕路。
        # 这里让楼梯列进入 Parkour 的那一格：vx 取正、vy=0、wz=0。
        # 其余列（离散方块、金字塔）完全不受影响，仍是全向指令——部署需要的横移/
        # 偏航能力在那些列上照常训练。
        # 【默认关闭】——路线 B 的架构论证依然成立（定向地形 + 全向指令是错配的），
        # 但它在本仓库里有一个致命的隐性副作用，实测已经烧掉两轮训练：
        #
        # N2_10dof_Env._reward_default_joint_pos（权重 1.0，健康 run 里 +0.719/秒、
        # 第三大正项）的最后两行是
        #     rew_filter = any(|commands[:,[1,2]]| > min_cmd_vel) or standing_cmd
        #     rew[rew_filter] = 1.0
        # 即"只要有横移或偏航指令（或站立），这一项直接白送满分"，只有在**纯前进**
        # 指令下才真去考核髋关节 yaw/roll 偏差。而 B 恰恰把楼梯列的 vy 和 wz 精确
        # 置零，于是那 30% 的环境从"白拿 1.0"变成"必须自己挣"。
        # 实测（零动作、1024 环境）：楼梯列拿到白送满分的比例 92% -> 6%，
        # 原始值 0.981 -> 0.772；未训练策略髋关节偏差大，实际损失更多。
        #
        # 健康 run 的每步净奖励只有 -0.335/秒，本来就贴着 only_positive_rewards 的
        # 截断线；抽掉这块正奖励、同时又强制那些环境必须真往前走，就把总和推过了
        # 截断线 -> 每步被钳成 0 -> 奖励梯度消失 -> 只剩熵项 -> noise_std 单调上涨
        # -> 摔得更多 -> 接触力尖峰 -> 更负。两个失败 run（0726_10-08-30_、
        # 0726_11-13-19_）都是这个形状：ep_len 恒在 10~16 步、noise_std 400 轮冲到 2.5，
        # 而同期健康 run 是 ep_len 24、noise_std 1.24。
        #
        # 要重新开启 B，必须先解决那个门控——例如在 N2PerceptiveEnv 里重写
        # _reward_default_joint_pos，让楼梯列的纯前进指令也享有同样的豁免；
        # 或者干脆走路线 A（对称地形），那样根本不需要 B。
        stairs_forward_only = False
        # 楼梯列上前进速度的重采样下限（m/s），区间 [stairs_min_vx, lin_vel_x_max]。
        # 0.3 -> 0.25：0.3 配 abs().clamp() 会在下限堆出质量点，等于从第 0 轮就
        # 强制 30% 的环境快走，实测 40 轮 ep_len 只有 5.9（关掉 forward_only 是 9.5）。
        # 取 0.25 而不是 0：低于 min_cmd_vel=0.2 的指令会被判成站立，把楼梯列的站立
        # 比例抬到近 30%，抵消 standing_prob 的下调。0.25 刚好在其上。
        # 原本设下限是担心"贴着楼梯磨蹭也算服从"，但路线 C 的课程只认真实 x 进展，
        # 磨蹭本来就升不了级，这个顾虑是多余的。
        stairs_min_vx = 0.25

    class domain_rand(N2_10dof_Cfg.domain_rand):
        # 关掉"每次 reset 重抽摩擦/恢复系数"。实测（1024 env）不含 reset 的步
        # 52.7ms、含 reset 的步 296.8ms，而训练早期 episode 只有十几步、几乎每步
        # 都有环境在 reset，于是 _refresh_actor_rigid_shape_props 的 per-env
        # Python 循环吃掉了大部分采样时间（服务器上表现为 1884 steps/s、52s 一轮）。
        # 建环境时的 per-env 摩擦随机化（_process_rigid_shape_props，上游 legged_gym
        # 的做法）依旧生效，4096 个环境仍各自持有取自 256 个 bucket 的不同摩擦，
        # 策略看到的摩擦分布不变；失去的只是"同一环境跨 episode 更换摩擦"。
        # 只覆盖 perceptive，盲策略保持原行为。
        refresh_shape_props_on_reset = False

    class noise(N2_10dof_Cfg.noise):
        class noise_scales(N2_10dof_Cfg.noise.noise_scales):
            height_measurements = 0.0       # privileged = 干净真值,必须为0

    class rewards(N2_10dof_Cfg.rewards):
        # ε : 旧离散实现的阈值（采样点低于支撑面超过该值即判为悬空）。
        # 已被 foothold_flat_k 的平滑形式取代，保留仅为记录当时的量纲参考。
        foothold_depth_tol = 0.04
        # _reward_foothold 的平滑系数 k：raw = 1 - exp(-k * 平均悬空深度)。
        # 取 20 使旧阈值 4cm 落在 1-exp(-0.8)=0.55 附近（区分度最大的中段），
        # 实测悬空深度中位 0.015m→0.26、p99 0.098m→0.86，动态范围用满。
        # k 越大越苛刻（对浅悬空也重罚），越小越宽容。
        foothold_flat_k = 20.0
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


        anti_freeze_speed = 0.15

        # 见 N2PerceptiveEnv._reward_feet_contact_forces（有界重写版）。
        # 300 -> 400 N：本机器人自重 33.2kg = 325 N，阈值卡在自重之下意味着单脚支撑期
        # （正常步态的一半时间）光是站住就在扣分——这也是此前对照实验里该项呈现
        # "地形无关"（平地 -4.85 / 楼梯 -5.0）的原因：它是恒定步态税，不是踩楼梯的代价。
        # 400 N ≈ 1.23 倍体重，正常行走基本免费，只有真正的硬冲击才被罚。
        max_contact_force = 400.
        # 每只脚超出部分的上限（N）。基类不封顶，摔倒砸地 |F| 上千牛时单步惩罚能盖过
        # 整个正项栈，把 only_positive_rewards 的截断变成"永远为 0"的死区。
        # 封在 200 N 后单步最坏 = 2 脚 x 200 x 0.05 x dt = -0.4，与正项同量级。
        feet_contact_force_max_excess = 200.

        world_progress_min_speed = 0.1

        class scales(N2_10dof_Cfg.rewards.scales):

            foothold = -0.5

            feet_air_time = 4.0

            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.8


            world_progress = 0.8
            world_heading = 1.0

            # 反冻结：命令要求前进却原地不动时把"动"抬到"冻"之上，破解楼梯前站死的
            # 局部最优（见 N2PerceptiveEnv._reward_anti_freeze）。正奖励、低速饱和，
            # 量级与 tracking_lin_vel(1.4) 同级即可，太大反而催生莽撞前冲。先给 1.0，
            # 跑一轮看 rew_anti_freeze 是否随 terrain_level 上升而上升、且 stand_still
            # 不反弹；若楼梯前仍犹豫可加到 1.5~2.0。
            anti_freeze = 1.0

            # 障碍物/楼梯通行相关：碰撞与踢竖面惩罚（原本已实现但未启用）
            # collision: 参考 legged_gym 上游 base 默认值及 anymal_c/a1 rough
            # terrain 配置（均未覆盖此值，直接沿用 -1.0 用于实际训练）
            collision = -1.0
            # stumble: 对应 _reward_stumble（注意 key 必须是 stumble，不是
            # legged_gym 里名字对不上的 feet_stumble，否则会 AttributeError）。
            # 上游没有任何参考配置启用过这一项，这里的数值是按 collision 同量级
            # 给的经验起点，需要在下一轮训练里看 TensorBoard 再调
            stumble = -2.0


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'
        # 高度图(96维,占每帧135维的71%)缩放后典型幅度比其余本体感知维度大很多,
        # 基类默认不做观测归一化(Identity),这里为 perceptive 单独打开经验归一化
        empirical_normalization = True