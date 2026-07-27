"""按 N2ParkourCfg.terrain 的参数生成 MuJoCo 测试赛道：上楼梯 -> 踏石 -> 下楼梯 -> 跨栏。
输出 MJCF geom 片段 + 与训练侧同规则的 goal 路点表。"""
import numpy as np
def step_goal_indices(num_steps, num_goals, power=2.0):
    # 与 humanoid/utils/terrain.py 同名函数逐行一致（此处内联，避免 import 拽进 isaacgym）
    num_steps = max(num_steps, num_goals)
    k = np.arange(1, num_goals + 1) / num_goals
    idx = np.maximum(np.round(num_steps * k ** power).astype(int), 1)
    for i in range(1, num_goals):
        idx[i] = max(idx[i], idx[i - 1] + 1)
    idx[-1] = num_steps
    for i in range(num_goals - 2, -1, -1):
        idx[i] = min(idx[i], idx[i + 1] - 1)
    return idx

D = 0.4                      # 统一难度档（训练里 difficulty = level/num_rows）
YHALF = 2.5                  # 赛道外缘半宽（沿用原 XML）
LANE_HW = 0.75               # 台阶通道半宽 = mean(parkour_half_valid_width)=(0.7+0.8)/2
TREAD = 0.30                 # mean(parkour_x_range)=(0.2+0.4)/2
RISER = 0.05 + D*0.15        # parkour_step_height_range [0.05,0.20] 线性插值 -> 0.11
RISER = 0.10                 # 取整到 0.10，与原 XML 楼梯一致、便于对比
NSTEP = 18                   # parkour_step_count
NGOAL = 8                    # num_goals=10 -> 8 个落在障碍上
PLAT = 2.5                   # parkour_platform_len
TRENCH = 0.30                # parkour_stepdown_outside_margin

S_LEN  = 0.9 + D*(0.45-0.9)  # parkour_stone_len_range   -> 0.72
S_WID  = 1.0 + D*(0.5-1.0)   # parkour_stone_width_range -> 0.80
S_GAP  = 0.05 + D*(0.20-0.05)# parkour_stone_gap_range   -> 0.11
S_Y    = 0.14                # parkour_stone_y_range (0.10,0.18) 取中
S_PIT  = 0.5                 # parkour_stone_pit_depth
NSTONE = 8

H_LEN  = 0.3                 # parkour_hurdle_len
H_SPACE= 1.5                 # parkour_hurdle_x_range (1.2,1.8) 取中
H_TOP  = 0.10 + D*(0.30-0.10)# parkour_hurdle_height_range -> 0.18
NHURD  = 8
# parkour_hurdle_y_range(-0.4,0.4) / half_valid_width(0.4,0.8)：给每道栏不同的横向
# 偏置与宽度，复刻训练里逐道随机的情形（固定值以保证测试可复现）
H_CY  = [ 0.00,-0.25, 0.30,-0.15, 0.20, 0.35,-0.30, 0.10]
H_HW  = [ 0.80, 0.60, 0.70, 0.50, 0.75, 0.55, 0.65, 0.45]

geoms=[]; goals=[]
def box(x0,x1,cy,hy,ztop,rgba=None,tag=""):
    """实心方块，从地面 0 长到 ztop（与原 XML 同风格）。"""
    if ztop<=1e-6: return
    c=' rgba="%s"'%rgba if rgba else ''
    geoms.append('      <geom type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f %.4f"'
                 ' friction="0.6 0.005 0.0001"%s/>%s'
                 %((x1-x0)/2, hy, ztop/2, (x0+x1)/2, cy, ztop/2, c, ('  <!-- %s -->'%tag) if tag else ''))

# ---------------- 段1 上楼梯 ----------------
x = PLAT/2 + 0.0                      # 机器人在 x=0，起步平台中点对齐 -> 第一级在 1.25m
geoms.append('      <!-- ===== 段1 上楼梯：%d 级 x 踏面%.2fm x 踢面%.2fm，通道宽%.1fm，'
             '通道外为 0 高度低地(地面) ===== -->'%(NSTEP,TREAD,RISER,2*LANE_HW))
up_goal_at = step_goal_indices(NSTEP,NGOAL,2.0)
up_slot={int(s):i for i,s in enumerate(up_goal_at)}
z=0.0
for k in range(1,NSTEP+1):
    x0,x1 = x+TREAD*(k-1), x+TREAD*k
    z = RISER*k
    box(x0,x1,0.0,LANE_HW,z,tag=("上楼 %d/%d"%(k,NSTEP)) if k in (1,NSTEP) else "")
    if k in up_slot: goals.append((round((x0+x1)/2,3),0.0,round(z,3)))
x_up_end = x+TREAD*NSTEP; z_top = z
# 平台A
PA=2.0
geoms.append('      <!-- 平台A：踏石段的起步平台 -->')
box(x_up_end, x_up_end+PA, 0.0, LANE_HW, z_top)
x = x_up_end+PA

# ---------------- 段2 踏石 ----------------
geoms.append('      <!-- ===== 段2 踏石：%d 块 %.2fx%.2fm，缝%.2fm，左右交替±%.2fm，'
             '坑深%.1fm ===== -->'%(NSTONE,S_LEN,S_WID,S_GAP,S_Y,S_PIT))
stone_x0 = x
pitch = S_GAP+S_LEN
stone_x1 = x + NSTONE*pitch
# 坑底：满幅平板，顶面比踏石低 pit_depth
box(stone_x0, stone_x1, 0.0, YHALF, z_top-S_PIT, rgba="0.35 0.30 0.28 1", tag="坑底(比踏石低 %.1fm)"%S_PIT)
side=1
for k in range(NSTONE):
    x0 = x + S_GAP + k*pitch; x1 = x0+S_LEN
    cy = side*S_Y; side=-side
    box(x0,x1,cy,S_WID/2,z_top,rgba="0.55 0.45 0.35 1",tag="踏石 %d"%(k+1))
    goals.append((round((x0+x1)/2,3),round(cy,3),round(z_top,3)))
x = stone_x1
# 平台B
PB=2.0
geoms.append('      <!-- 平台B：下楼梯的起步平台 -->')
box(x,x+PB,0.0,LANE_HW,z_top); x+=PB

# ---------------- 段3 下楼梯 ----------------
geoms.append('      <!-- ===== 段3 下楼梯：%d 级，通道外沿 x 跟着一起降、恒低 %.1fm(平行沟)，'
             '低于地面的部分由地面平面兜底 ===== -->'%(NSTEP,TRENCH))
dn_slot={int(s):i for i,s in enumerate(step_goal_indices(NSTEP,NGOAL,2.0))}
for k in range(1,NSTEP+1):
    x0,x1 = x+TREAD*(k-1), x+TREAD*k
    zl = z_top - RISER*k
    box(x0,x1,0.0,YHALF, zl-TRENCH, rgba="0.30 0.30 0.34 1")    # 满幅沟底(低于通道 0.3m)
    box(x0,x1,0.0,LANE_HW, zl, tag=("下楼 %d/%d"%(k,NSTEP)) if k in (1,NSTEP) else "")
    if k in dn_slot: goals.append((round((x0+x1)/2,3),0.0,round(zl,3)))
x = x+TREAD*NSTEP; z_bot = z_top-RISER*NSTEP
assert abs(z_bot)<1e-9, z_bot

# ---------------- 段4 跨栏 ----------------
PC=2.0; x+=PC
geoms.append('      <!-- ===== 段4 跨栏：%d 道，厚%.2fm 高%.2fm，间距%.1fm，'
             '横向偏置与宽度逐道不同；栏两侧是平地(EP 原样，靠 goal 而非墙防绕行) ===== -->'%(NHURD,H_LEN,H_TOP,H_SPACE))
for k in range(NHURD):
    cx = x + H_SPACE*(k+1)
    box(cx-H_LEN/2, cx+H_LEN/2, H_CY[k], H_HW[k], H_TOP, rgba="0.75 0.35 0.25 1", tag="跨栏 %d"%(k+1))
    goals.append((round(cx-H_SPACE/2,3), round(H_CY[k],3), 0.0))   # goal 落在两栏之间(EP 原样)
x_end = x + H_SPACE*NHURD
goals.append((round(x_end+1.0,3),0.0,0.0))

print("\n".join(geoms))
print("\n<!-- 赛道总长 %.2f m，最高点 %.2f m，goal 共 %d 个 -->"%(x_end+1.0, z_top, len(goals)))
import json
open('/home/jun/.claude/jobs/3ab4d3fd/tmp/course_goals.json','w').write(json.dumps(goals))
