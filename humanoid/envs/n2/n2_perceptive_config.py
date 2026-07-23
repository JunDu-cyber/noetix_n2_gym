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
        terrain_proportions = [0.1, 0.0, 0.1, 0.05, 0.05, 0.4, 0.3]

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

        class scales(N2_10dof_Cfg.rewards.scales):
            foothold = -0.15  # sign lives here; reward fn returns +count

            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.6

            # 障碍物/楼梯通行相关：碰撞与踢竖面惩罚（原本已实现但未启用）
            # collision: 参考 legged_gym 上游 base 默认值及 anymal_c/a1 rough
            # terrain 配置（均未覆盖此值，直接沿用 -1.0 用于实际训练）
            collision = -1.0
            # stumble: 对应 _reward_stumble（注意 key 必须是 stumble，不是
            # legged_gym 里名字对不上的 feet_stumble，否则会 AttributeError）。
            # 上游没有任何参考配置启用过这一项，这里的数值是按 collision 同量级
            # 给的经验起点，需要在下一轮训练里看 TensorBoard 再调
            stumble = -1.0


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'
        # 高度图(96维,占每帧135维的71%)缩放后典型幅度比其余本体感知维度大很多,
        # 基类默认不做观测归一化(Identity),这里为 perceptive 单独打开经验归一化
        empirical_normalization = True