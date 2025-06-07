import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Polygon1
class Params:
    def __init__(self):
        self.world_bounds_x = [0, 200]  # [m], map size in X-direction
        self.world_bounds_y = [0, 200]  # [m], map size in Y-direction
        self.drone_vel = 2  # velocity of UAV
        self.acc = 3  # 5, acceleration，1m/s2
        self.w_vel = math.pi  # rad/s
        self.dt = 0.1
        self.influence_radius_obs = 6  # 8 potential fields radius
        self.goal_tolerance = 0.8  # [m], maximum distance threshold to reach the goal
        self.num_robots = 15  # number of robots in the formation
        self.interrobots_dist = 10  # [m], distance between robots in default formation
        self.min_safe_dis = 0.25
        self.repulsive_coef = 10
        self.attractive_coef = 0.1
        self.ali_param = 0.1  # 大一些就会及时预知并避撞
        self.coh = 0.01
        self.min_between_obs = 5

def draw_map(obstacles, params,ax):
    # Bounds on world
    world_bounds_x = [params.world_bounds_x[0], params.world_bounds_x[1]]
    world_bounds_y = [params.world_bounds_y[0], params.world_bounds_y[1]]
    # Draw obstacles
    #ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(world_bounds_x)
    ax.set_ylim(world_bounds_y)
    for k in range(len(obstacles)):
        ax.add_patch(Polygon1(obstacles[k], color='#777777', zorder=10))
    plt.axis('on')  # 启用坐标轴
    # 设置刻度间隔
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
    #ax.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    #ax.tick_params(axis='both', which='major', labelsize=10)  # 设置刻度字体大小
    #ax.set_xlabel('X-axis (m)', fontsize=10)  # 设置 x 轴标签
    #ax.set_ylabel('Y-axis (m)', fontsize=10)  # 设置 y 轴标签
    #ax.set_title('Multi-Agent Formation Control', fontsize=16)  # 设置标题

# 读取障碍物信息
obstacles = []
with open(r'./result/obstacles_rec.txt', 'r') as file:
    for line in file:
        obstacles.append(eval(line.strip()))  # 使用 eval 将字符串转换为列表

# 读取所有机器人的轨迹信息
import json
import numpy as np
with open(r'./result/trajectory_rec.json', 'r') as file:
    trajectory = json.load(file)
with open(r'./result/compare_trajectory_rec.json', 'r') as file:
    compare_trajectory = json.load(file)
with open(r'./result/xiaorong_trajectory_rec.json', 'r') as file:
    xiaorong_trajectory = json.load(file)
# 手动将字符串键转换为整数
trajectory = {int(k): np.array(v) for k, v in trajectory.items()}  # 再次将列表转换为 NumPy 数组

params = Params()

#colors = ['red', 'green', 'skyblue', 'blue', 'magenta', 'yellow', 'gray', 'purple', 'olive', 'teal', 'pink', 'palegreen', 'coral', 'turquoise', 'navy']
colors = ['#e84445', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2']
UAVs = ['UAV$_{1}$', 'UAV$_{2}$', 'UAV$_{3}$', 'UAV$_{4}$', 'UAV$_{5}$', 'UAV$_{6}$', 'UAV$_{7}$', 'UAV$_{8}$', 'UAV$_{9}$', 'UAV$_{10}$', 'UAV$_{11}$', 'UAV$_{12}$', 'UAV$_{13}$', 'UAV$_{14}$', 'UAV$_{15}$']


# 创建图形和子图
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

# 绘制第一个子图
draw_map(obstacles, params,axs[0])
#初始时刻
t=0
for i in range(params.num_robots):
    axs[0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
R=11
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[0].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
axs[0].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#中间时刻1
t=int(len(trajectory[0])/8)
for i in range(params.num_robots):
    axs[0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

#中间时刻2
t=int(len(trajectory[0])/4)+100
for i in range(params.num_robots):
    axs[0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

axs[0].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#777777', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[0].legend(
    loc='upper right',      # 设置图例位置为右上角
    bbox_to_anchor=(1.01, 1.01),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
axs[0].set_title('BPF-FOA', fontsize=16)
axs[0].set_xlabel('x-axis', fontsize=16)
axs[0].set_ylabel('y-axis', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=16)
# 绘制第二个子图
draw_map(obstacles, params,axs[1])
trajectory = {int(k): np.array(v) for k, v in compare_trajectory.items()}  # 再次将列表转换为 NumPy 数组
#初始时刻
t=0
for i in range(params.num_robots):
    axs[1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[1].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
axs[1].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#中间时刻1
t=int(len(trajectory[0])/4)-100
for i in range(params.num_robots):
    axs[1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

#中间时刻2
t=int(len(trajectory[0])/2)-100
for i in range(params.num_robots):
    axs[1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线


axs[1].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#777777', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[1].legend(
    loc='upper right',      # 设置图例位置为右上角
    bbox_to_anchor=(1.01, 1.01),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
axs[1].set_title('CPF-OA', fontsize=16)
axs[1].set_xlabel('x-axis', fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=16)
# 绘制第三个子图
draw_map(obstacles, params,axs[2])
trajectory = {int(k): np.array(v) for k, v in xiaorong_trajectory.items()}  # 再次将列表转换为 NumPy 数组
#初始时刻
t=0
for i in range(params.num_robots):
    axs[2].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[2].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[2].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[2].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
axs[2].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[2].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#中间时刻1
t=int(len(trajectory[0])/8)+200
for i in range(params.num_robots):
    axs[2].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[2].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

#中间时刻2
t=int(len(trajectory[0])/4)+350
for i in range(params.num_robots):
    axs[2].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[2].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

axs[2].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#777777', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[2].legend(
    loc='upper right',      # 设置图例位置为右上角
    bbox_to_anchor=(1.01, 1.01),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
axs[2].set_title('BPF-OA', fontsize=16)
axs[2].set_xlabel('x-axis', fontsize=16)
axs[2].tick_params(axis='both', which='major', labelsize=16)
# 调整布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.1 )
plt.savefig('fig_trajectory_rec.jpg', dpi=600)  # 设置保存的分辨率

plt.show()