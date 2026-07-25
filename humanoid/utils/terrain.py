# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


import numpy as np

from isaacgym import terrain_utils
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

def add_roughness(terrain, noise_magnitude=0.02):
    terrain_utils.random_uniform_terrain(
        terrain,
        min_height=-noise_magnitude,
        max_height=noise_magnitude,
        step=0.005,
        downsampled_scale=0.075,
    )

def directional_stairs(terrain, step_width, step_height, platform_size=1.5):
    """-x 端底部平台 + 沿 +x 单向逐级升/降、贯穿整块地形的定向长楼梯（沿 y 恒定）。

    机器人在 -x 端的底部平台出生（见 N2PerceptiveEnv._reset_root_states 把楼梯格的
    出生点挪到这里），正对一整块地形长度的楼梯往 +x 爬(up)或走下(down)。

    【为什么必须有这块平台】isaacgym 的 terrain_utils.stairs_terrain 是整个地形族里
    唯一不接受 platform_size 的函数——legged_gym 的每个子地形都传 platform_size=3./4.，
    IsaacLab 的 pyramid_stairs 用 platform_width=3.0，都是为了保证"机器人出生在平地上"。
    而出生高度沿用 legged_gym 的 env_origin_z = max(中心±1m 窗口)，该式默认中心是平的；
    在单调楼梯上它会取到前方 1m 处的最高台阶，实测出生点悬空：台阶 5cm→0.20m、
    10cm→0.40m、20cm→0.80m（金字塔楼梯因为有平台恒为 0.00m）。也就是每次 reset 都在
    自由落体。加上这块平台后，出生点落在 0 高度的平地上，契约恢复。

    -x 端(轴 0 低端)紧贴的是上一难度级的同类楼梯，其顶比本块的底高，天然形成一堵
    背墙，正好挡住后退、逼机器人正面往上爬。沿 y 恒定=定向：不能沿 y 等高绕圈
    （pyramid 沿 xy 都收所以能绕，这里只沿 x 收；y 向等高绕行由世界系奖励压制，
    实测最新策略横移已降到 ~0.3m）。

    step_height>0 -> 往 +x 升高（底部平台在最低，前进=上楼，index 7）；
    step_height<0 -> 往 +x 降低（底部平台在最高，前进=下楼，index 8）。
    平台恒为 0 高度；_reset_root_states 据此把出生 z 设成"平台高度+站立高度"。
    """
    sw = max(1, int(step_width / terrain.horizontal_scale))
    sh = int(step_height / terrain.vertical_scale)
    plat = max(1, int(platform_size / terrain.horizontal_scale))
    n = terrain.height_field_raw.shape[0]
    x = np.arange(n)
    d = x - plat                                    # 超出底部平台多少像素（<=0 即平台内）
    k = np.where(d <= 0, 0, (d + sw - 1) // sw)     # 第几级台阶（平台=0，往 +x 递增）
    terrain.height_field_raw[:, :] = (k * sh)[:, None]

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.vertical_scale = cfg.vertical_scale
        self.horizontal_scale = cfg.horizontal_scale

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

class HumanoidTerrain(Terrain):
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        super().__init__(cfg, num_robots)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        step_width = np.random.uniform(0.3, 0.45)
        discrete_obstacles_height = difficulty * 0.20
        r_height = difficulty * 0.07 # 0.07
        h_slope = difficulty * 0.15 # 0.15
        if choice < self.proportions[0]:
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[1]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[2]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-r_height, max_height=r_height, step=0.005, downsampled_scale=0.2)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[3]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[4]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=-h_slope, platform_size=0.1)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[5]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[6]:
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=-discrete_obstacles_height, platform_size=1.)
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[7]:
            directional_stairs(terrain, step_width, discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        elif choice < self.proportions[8]:
            directional_stairs(terrain, step_width, -discrete_obstacles_height,
                               getattr(self.cfg, 'stairs_platform_size', 1.5))
            add_roughness(terrain, np.random.uniform(0.01, 0.05))
        else:
            pass
        return terrain
