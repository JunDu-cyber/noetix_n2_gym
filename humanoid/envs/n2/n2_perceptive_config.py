from humanoid.envs.n2.n2_config import N2_18DofCfg, N2_18DofCfgPPO

class N2PerceptiveCfg(N2_18DofCfg):
    class env(N2_18DofCfg.env):
        # 96 = len(measured_points_x)=12 × len(measured_points_y)=8
        num_observations = 63 + 96          # 159
        # 如果 N2Cfg 里有 num_privileged_obs(asymmetric critic),同样 +96

    class terrain(N2_18DofCfg.terrain):
        measure_heights = True              # blind 里本来也是 True,显式写上保险
        curriculum = True

    class noise(N2_18DofCfg.noise):
        class noise_scales(N2_18DofCfg.noise.noise_scales):
            height_measurements = 0.0       # privileged = 干净真值,必须为0

class N2PerceptiveCfgPPO(N2_18DofCfgPPO):
    class runner(N2_18DofCfgPPO.runner):
        experiment_name = 'n2_perceptive'