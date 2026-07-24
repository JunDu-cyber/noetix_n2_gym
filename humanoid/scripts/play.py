# 导入操作系统相关功能
import os
# 导入系统相关功能
import sys
# 从humanoid模块导入根目录路径
from humanoid import LEGGED_GYM_ROOT_DIR

# 导入Isaac Gym库
import isaacgym
# 导入所有环境相关模块
from humanoid.envs import *
# 导入工具函数和类
from humanoid.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

# 导入数值计算库
import numpy as np
# 导入PyTorch深度学习框架
import torch


def play(args):
    """
    播放/测试函数：加载训练好的策略模型并在环境中运行以可视化结果
    
    参数:
        args: 命令行参数对象，包含运行所需的各种配置
    """
    # 获取环境和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 重写一些测试参数
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 限制环境数量为1
    env_cfg.env.num_envs = 1  # 设置环境数量为1
    env_cfg.sim.physx.max_gpu_contact_pairs = 2**10  # 设置GPU接触对的最大数量
    env_cfg.terrain.mesh_type = 'trimesh'  # 设置地形类型为
    env_cfg.terrain.num_rows = 8  # 设置地形行数
    env_cfg.terrain.num_cols = 8  # 设置地形列数
    env_cfg.terrain.curriculum = False  # 课程学习
    env_cfg.terrain.max_init_terrain_level = 5
    # 9 项，与 HumanoidTerrain.make_terrain 的新增 straight stairs 分支对齐
    # （7 项时 cumsum 在 index 6 就到 1.0，永远走不到 index 7/8，看不到新地形）
    env_cfg.terrain.terrain_proportions = [0., 0.0, 0.1, 0.0, 0.0, 0.05, 0.2, 0.2, 0.15]
    # env_cfg.terrain.selected = True
    # env_cfg.terrain.terrain_kwargs = {'type': 'pyramid_stairs_terrain',
    #                                   'step_width': 0.30,
    #                                   'step_height': 0.16,   # 分三次改这个数
    #                                   'platform_size': 2.}
    env_cfg.noise.add_noise = False  # 关闭噪声添加
    env_cfg.domain_rand.randomize_gains = False  # 关闭增益随机化
    env_cfg.domain_rand.randomize_motor_strength = False  # 关闭电机强度随机化
    env_cfg.domain_rand.randomize_base_mass = False  # 关闭基础质量随机化
    env_cfg.domain_rand.randomize_com_displacement = False  # 关闭质心位移随机化
    env_cfg.domain_rand.randomize_friction = False  # 关闭摩擦系数随机化
    env_cfg.domain_rand.push_robots = False  # 关闭机器人推动
    env_cfg.domain_rand.disturbance = False  # 关闭干扰
    env_cfg.domain_rand.disturbance_probabilities = 0.005  # 设置干扰概率
    env_cfg.domain_rand.push_force_range = [50.0, 500.0]  # 设置推力范围
    env_cfg.domain_rand.push_torque_range = [0.0, 0.0]  # 设置扭矩范围
    env_cfg.env.episode_length_s = 100  # 设置每轮episode的时长（秒）

    # 设置为测试模式
    env_cfg.env.test = True

    # 如果控制机器人标志为真，则进一步调整参数
    if CONTROL_ROBOT:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # 确保环境数量不超过1
        env_cfg.env.episode_length_s = 100  # 设置episode时长
        env_cfg.commands.resampling_time = [1000, 1001]  # 设置命令重采样时间
        env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]  # 设置x方向线速度范围
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]  # 设置y方向线速度范围
        env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]  # 设置偏航角速度范围

    # 准备环境
    # env: 环境对象，用于模拟和交互
    # _: 忽略第二个返回值
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 获取初始观测值
    obs = env.get_observations()
    
    # 加载策略
    train_cfg.runner.resume = True  # 设置为恢复模式
    # 创建算法运行器实例
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # 获取推理策略
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 将策略导出为JIT模块（用于C++中运行）
    if EXPORT_POLICY:
        # 构建导出路径
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        # 导出策略为JIT模块
        export_policy_as_jit(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer)
        # 导出策略为ONNX格式
        export_policy_as_onnx(ppo_runner.alg.policy, path, ppo_runner.obs_normalizer)
        print('Exported policy to: ', path)

    # 创建日志记录器
    logger = Logger(env)
    robot_index = 0  # 用于日志记录的机器人索引
    joint_index = 1  # 用于日志记录的关节索引
    stop_state_log = 100  # 开始绘制状态前的步数
    stop_rew_log = env.max_episode_length + 1  # 开始打印平均奖励前的步数
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)  # 相机位置
    camera_vel = np.array([1., 1., 0.])  # 相机速度
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)  # 相机方向
    img_idx = 0  # 图像索引

    # 主循环：运行指定次数的episode
    for i in range(10*int(env.max_episode_length)):
        # 获取策略动作
        actions = policy(obs.detach())
        # 执行动作并获取新的状态
        # obs: 新的观测值
        # _: 特权观测值（此处忽略）
        # rews: 奖励值
        # dones: 完成标记
        # infos: 额外信息
        # _: 终止ID（此处忽略）
        # _: 终止特权观测值（此处忽略）
        env.commands[:,0] = 1.0  # 控制x方向线速度为1.0
        env.commands[:,1] = 0.0  # 控制y方向线速度为0.0
        env.commands[:,2] = 0.0  # 控制偏航角速度
        obs, _, rews, dones, infos,_,_ = env.step(actions.detach())
        
        # 如果需要录制帧，则保存图像
        if RECORD_FRAMES:
            if i % 2:  # 每隔一步保存一次图像
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
                
        # 如果需要移动相机，则更新相机位置
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # 记录状态日志
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,  # 目标关节位置
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),  # 实际关节位置
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),  # 关节速度
                    'dof_torque': env.torques[robot_index, joint_index].item(),  # 关节扭矩
                    'command_x': env.commands[robot_index, 0].item(),  # x方向命令
                    'command_y': env.commands[robot_index, 1].item(),  # y方向命令
                    'command_yaw': env.commands[robot_index, 2].item(),  # 偏航命令
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),  # 基座x方向速度
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),  # 基座y方向速度
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),  # 基座z方向速度
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),  # 基座偏航角速度
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()  # 接触力z分量
                }
            )
        # 如果达到记录步数，则绘制状态图
        # elif i==stop_state_log:
        #     logger.plot_states()
            
        # 记录奖励日志
        if  0 < i < stop_rew_log:
            if infos["episode"]:  # 如果有episode信息
                num_episodes = torch.sum(env.reset_buf).item()  # 计算完成的episode数量
                if num_episodes>0:  # 如果有完成的episode
                    logger.log_rewards(infos["episode"], num_episodes)  # 记录奖励
                    
        # 如果达到奖励记录步数，则打印奖励
        elif i==stop_rew_log:
            logger.print_rewards()
   
# 程序入口点
if __name__ == '__main__':
    # 设置全局标志
    EXPORT_POLICY = True  # 是否导出策略
    CONTROL_ROBOT = False  # 是否控制机器人
    RECORD_FRAMES = False  # 是否录制帧
    MOVE_CAMERA = False  # 是否移动相机

    # 获取命令行参数
    args = get_args()
    args.num_envs = 1  # 设置环境数量为1
    # 设置加载运行的路径
    #args.load_run = "/home/yue/noetix_n2_gym_test/noetix_n2_gym/logs/n2/1111_19-05-45_"
    args.checkpoint = -1  # 设置检查点为-1（表示最新）
    
    # 调用播放函数
    play(args)