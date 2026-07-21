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


class N2PerceptiveCfgPPO(N2_10dof_CfgPPO):
    class runner(N2_10dof_CfgPPO.runner):
        experiment_name = 'n2_perceptive'