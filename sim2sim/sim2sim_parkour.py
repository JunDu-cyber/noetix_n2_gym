"""n2_parkour 的 MuJoCo sim2sim。

与 sim2sim_perceptive.py 的区别只有一处，但很关键：**Parkour 策略的目标是 goal
路点的位置，不是速度指令的方向**，观测里多了到当前 goal 的 [cos dpsi, sin dpsi]
两维（num_single_obs 135 -> 137）。所以这里必须自己维护一套 goal 路点，并复刻
N2ParkourEnv._update_goals 的推进逻辑。

【重要前提，先说清楚】
Isaac Gym 里 goal 是地形生成时就知道的真值，MuJoCo 这边没有这个"上帝视角"。
本脚本从场景几何里推算路点（沿 +x 每隔 goal_spacing 取一个，z 用向下射线打到的
地形高度），目的是**验证策略本身**——它到底会不会朝目标爬楼梯。
这不是一条部署路径：真机上没有任何东西会告诉机器人 goal 在哪。Extreme Parkour
自己的解法是再训一个以深度相机为输入的学生策略去蒸馏这个教师策略，本仓库尚未实现。
所以：本脚本用于评估，不要据此认为 parkour 策略已经可部署。

其余部分（PD 控制、高度扫描、观测缩放、帧堆叠）与 perceptive 完全一致，直接复用
sim2sim_perceptive 里的实现，避免两份代码各自漂移。
"""
import math
import copy
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from humanoid import LEGGED_GYM_ROOT_DIR
import torch
from pynput.keyboard import Listener, Key
import yaml

# 复用 perceptive 侧已经调通的实现：观测提取、PD、高度扫描射线、键盘指令
from sim2sim_perceptive import (
    cmd, get_obs, pd_control, init_height_points,
    terrain_height_at, get_height_scan, get_height_points_world,
)


class Carrot:
    """
    按键：
      ← / →      offset 的世界系 y 增/减（转向）
      ↑ / ↓      offset 的世界系 x 增/减（拉远/拉近）
      Home/End   整体拉远 / 拉近（沿当前 offset 方向缩放）
      Insert/F1  把 offset 重置为"当前朝向正前方 lookahead 米"（只取一次 yaw，
                 此后就是世界系常量，不再跟着转）
    """

    def __init__(self, lookahead=1.5, step=0.15):
        self.offset = None          # 世界系 (dx, dy)，相对机器人当前位置
        self.lookahead = lookahead
        self.step = step
        self._pending = []          # 待处理按键，避免回调线程直接改状态

    def on_press(self, key):
        self._pending.append(key)

    def _apply(self, yaw):
        while self._pending:
            k = self._pending.pop(0)
            if self.offset is None:
                self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
            if k == Key.left:      self.offset[1] += self.step
            elif k == Key.right:   self.offset[1] -= self.step
            elif k == Key.up:      self.offset[0] += self.step
            elif k == Key.down:    self.offset[0] -= self.step
            elif k in (Key.home, Key.end):
                n = np.linalg.norm(self.offset)
                if n > 1e-6:
                    scale = (n + (0.1 if k == Key.home else -0.1)) / n
                    self.offset = self.offset * max(scale, 0.05)
            elif k in (Key.insert, Key.f1):
                # 只在此刻取一次 yaw，之后 offset 就是世界系常量
                self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
            else:
                continue
            print("[carrot] offset=(%+.2f, %+.2f) 世界系  |offset|=%.2f m"
                  % (self.offset[0], self.offset[1], np.linalg.norm(self.offset)))

    def update(self, base_xy, yaw, reach, model, data):
        """返回胡萝卜的世界坐标 (x, y, z)。reach 参数保留只为签名统一，此模式不用。"""
        if self.offset is None:
            self.offset = np.array([math.cos(yaw), math.sin(yaw)]) * self.lookahead
        self._apply(yaw)
        pos = base_xy + self.offset
        z = terrain_height_at(model, data, pos[None, :])[0]
        return np.array([pos[0], pos[1], z])


def build_goals(model, data, start_xy, cfg):
    """生成 goal 路点。

    优先用 yaml 里显式给的 goals（世界坐标 [[x,y],...]）；没有就沿 +x 自动铺：
    从 goal_first_x 起每隔 goal_spacing 一个，y 固定为出生点的 y，z 用向下射线
    取该处地形高度——等价于"把 goal 放在台阶顶面上"，这正是 Extreme Parkour 的
    parkour_step_terrain 干的事（goals[i+1] 落在第 i 级障碍上）。
    """
    explicit = cfg.get("goals")
    if explicit:
        pts = np.array(explicit, dtype=np.float64)
        if pts.shape[1] == 2:                       # 只给了 xy，z 用射线补
            z = terrain_height_at(model, data, pts)
            pts = np.concatenate([pts, z[:, None]], axis=1)
        return pts

    first = float(cfg.get("goal_first_x", 1.0))
    spacing = float(cfg.get("goal_spacing", 0.25))
    count = int(cfg.get("goal_count", 20))
    xs = first + spacing * np.arange(count)
    ys = np.full_like(xs, start_xy[1])
    z = terrain_height_at(model, data, np.stack([xs, ys], axis=-1))
    return np.stack([xs, ys, z], axis=-1)


def run_mujoco(cfg_name, command, carrot=None):
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{cfg_name}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    num_single_obs = config["num_single_obs"]
    frame_stack = config["frame_stack"]
    height_measurements_scale = config["height_measurements_scale"]
    base_height_offset = config["base_height_offset"]
    height_clip = config["height_clip"]
    measured_points_x = config["measured_points_x"]
    measured_points_y = config["measured_points_y"]
    goal_reach_dist = float(config.get("goal_reach_dist", 0.5))
    goal_mode = config.get("goal_mode", "waypoints")   # waypoints | carrot
    debug_viz = bool(config.get("debug_viz", True))
    tau_limit = np.array(config["tau_limit"], dtype=np.float32) if "tau_limit" in config else None

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)
    policy = torch.jit.load(policy_path)

    height_points = init_height_points(measured_points_x, measured_points_y)
    num_height_points = height_points.shape[0]
    # 137 = 9(cmd+angvel+gravity) + 3*num_actions + 2(goal) + 高度点数
    expect = 9 + num_actions * 3 + 2 + num_height_points
    assert num_single_obs == expect, (
        f"num_single_obs ({num_single_obs}) != 9 + 3*{num_actions} + 2(goal) + "
        f"{num_height_points}(height) = {expect}")
    height_marker_world = np.zeros((num_height_points, 3))

    default_dof_pos = default_angles
    data.qpos[7:] = default_dof_pos
    mujoco.mj_step(model, data)

    if goal_mode == "carrot":
        goals = None
        cur_goal = 0
        print("[parkour] goal_mode=carrot —— 用方向键牵引目标点")
        print("          <-/->  左右挪   ^/v  前后挪   Home/End  调 lookahead   Insert/F1  对齐正前方")
    else:
        goals = build_goals(model, data, data.qpos[:2].copy(), config)
        cur_goal = 0
        print(f"[parkour] goal_mode=waypoints —— {len(goals)} 个路点，x 从 {goals[0,0]:.2f} 到 {goals[-1,0]:.2f} m")
    print(f"[parkour] 到达半径 {goal_reach_dist} m")

    viewer = mujoco_viewer.MujocoViewer(model, data)
    target_q = np.zeros(num_actions, dtype=np.double)
    action = np.zeros(num_actions, dtype=np.double)
    hist_obs = deque([np.zeros([1, num_single_obs], dtype=np.double) for _ in range(frame_stack)],
                     maxlen=frame_stack)
    count_lowlevel = 0
    reached = 0

    for step in tqdm(range(int(simulation_duration / simulation_dt)), desc="Simulating..."):
        q, dq, quat, v, omega, gvec, base_xyz = get_obs(data)
        q = q[-num_actions:]
        dq = dq[-num_actions:]

        if count_lowlevel % control_decimation == 0:
            # 自身偏航（与 get_height_scan 里一致：仅取 yaw 分量）
            yaw = 2.0 * math.atan2(quat[2], quat[3])
            if goal_mode == "carrot":
                cur_xyz = carrot.update(base_xyz[:2].copy(), yaw, goal_reach_dist, model, data)
                to_goal = cur_xyz[:2] - base_xyz[:2]
            else:
                # ---- goal 推进：复刻 N2ParkourEnv._update_goals ----
                # 【两个条件缺一不可】此前这里只有 (a)，靠 goal_reach_dist=0.5 的大圈
                # 兜着才没出事；把半径对齐训练值 0.15 之后立刻暴露：机器人擦过 goal
                # 而没进圈，goal 留在身后，to_goal 掉头指向后方，Δψ 变成 ~180°，
                # 策略掉头往回走，整条通道就走不下去。训练侧 _update_goals 的注释
                # 原样预言了这个故障，只是当时没搬过来。
                #   (a) 进到 reach 半径内；
                #   (b) 已【越过】该 goal(x 为负)且横向没跑出通道(|dy| < lateral_tol)。
                to_goal = goals[cur_goal, :2] - base_xyz[:2]
                passed = to_goal[0] < 0 and abs(to_goal[1]) < goal_pass_lateral_tol
                if (np.linalg.norm(to_goal) < goal_reach_dist or passed) \
                        and cur_goal < len(goals) - 1:
                    cur_goal += 1
                    reached += 1
                    to_goal = goals[cur_goal, :2] - base_xyz[:2]
                cur_xyz = goals[cur_goal]
            target_yaw = math.atan2(to_goal[1], to_goal[0])
            dpsi = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))

            obs = np.zeros([1, num_single_obs], dtype=np.float32)
            obs[0, :3] = command.cmd * cmd_scale
            obs[0, 3:6] = omega * ang_vel_scale
            obs[0, 6:9] = gvec[:3]
            obs[0, 9:9 + num_actions] = (q - default_dof_pos) * dof_pos_scale
            obs[0, 9 + num_actions:9 + num_actions * 2] = dq * dof_vel_scale
            obs[0, 9 + num_actions * 2:9 + num_actions * 3] = action
            # goal 两维紧跟在 actions 之后，与 N2ParkourEnv.compute_observations 的顺序一致
            obs[0, 9 + num_actions * 3]     = math.cos(dpsi)
            obs[0, 9 + num_actions * 3 + 1] = math.sin(dpsi)
            obs[0, 9 + num_actions * 3 + 2:] = get_height_scan(
                model, data, base_xyz, quat, height_points,
                base_height_offset, height_clip, height_measurements_scale)

            if debug_viz:
                height_marker_world[:] = get_height_points_world(
                    model, data, base_xyz, quat, height_points)

            hist_obs.append(obs)
            model_input = np.zeros([1, num_obs], dtype=np.float32)
            for i in range(frame_stack):
                model_input[0, i * num_single_obs:(i + 1) * num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(model_input))[0].detach().numpy()
            target_q = action * action_scale + default_dof_pos

            if step % 500 == 0:
                tgt = ("carrot(%.2f,%.2f)" % (cur_xyz[0], cur_xyz[1])) if goal_mode == "carrot" \
                      else ("goal %d/%d" % (cur_goal, len(goals) - 1))
                print(f"  x={base_xyz[0]:.2f} z={base_xyz[2]:.2f} {tgt} "
                      f"dpsi={math.degrees(dpsi):+.0f}deg vx={v[0]:+.2f}")

        tau = pd_control(target_q, q, kps, np.zeros(num_actions, dtype=np.double), dq, kds)
        if tau_limit is not None:
            tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)

        if debug_viz:
            for px, py, pz in height_marker_world:
                viewer.add_marker(pos=[px, py, pz], size=[0.02, 0.02, 0.02],
                                  rgba=[1, 1, 0, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
            # 目标可视化：红色大球=当前目标（与 play.py 配色一致），绿色小球=后续路点
            viewer.add_marker(pos=[cur_xyz[0], cur_xyz[1], cur_xyz[2] + 0.08],
                              size=[0.12, 0.12, 0.12], rgba=[1, 0, 0, 1],
                              type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
            if goals is not None:
                for g in goals[cur_goal + 1:]:
                    viewer.add_marker(pos=[g[0], g[1], g[2] + 0.08], size=[0.05, 0.05, 0.05],
                                      rgba=[0, 1, 0, 0.6], type=mujoco.mjtGeom.mjGEOM_SPHERE, label="")
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    tail = f"到达 {reached} 个 goal，" if goals is not None else ""
    print(f"\n[parkour] 结束：{tail}最终 x={data.qpos[0]:.2f} z={data.qpos[2]:.2f}")

#TODO: add more terrains and align height map
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="n2_parkour.yaml",
                        help="config file name in sim2sim/configs")
    args = parser.parse_args()
    with open(f"{LEGGED_GYM_ROOT_DIR}/sim2sim/configs/{args.config_file}") as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    mode = _cfg.get("goal_mode", "waypoints")

    command = cmd()
    # parkour 训练时 vy/wz 恒为 0，前进速度固定给训练范围上限；方向由目标点决定
    command.cmd[:] = [float(_cfg.get("ranges_lin_vel_x_max", 0.8)), 0.0, 0.0]

    carrot = None
    if mode == "carrot":
        carrot = Carrot(lookahead=float(_cfg.get("carrot_lookahead", 1.5)),
                        step=float(_cfg.get("carrot_step", 0.15)))
        listener = Listener(on_press=carrot.on_press)      # 方向键牵胡萝卜
    else:
        listener = Listener(on_press=command.cmd_swtich)   # 方向键调速度指令
    listener.start()
    run_mujoco(args.config_file, command, carrot)
