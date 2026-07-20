import torch
from humanoid.envs.n2.n2_env import N2Env

class N2PerceptiveEnv(N2Env):
    def compute_observations(self):
        super().compute_observations()      # 先拼好原63维到 self.obs_buf
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1, 1.) * self.obs_scales.height_measurements
        self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)