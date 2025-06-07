import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Polygon1
from matplotlib.patches import Circle
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
    world_bounds_x =[80,200] #[params.world_bounds_x[0], params.world_bounds_x[1]]
    world_bounds_y = [-5,140]#[params.world_bounds_y[0], params.world_bounds_y[1]]
    # Draw obstacles
    #ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(world_bounds_x)
    ax.set_ylim(world_bounds_y)
    for k in range(len(obstacles)):
        ax.add_patch(Polygon1(obstacles[k], color='#777777'))
    plt.axis('on')  # 启用坐标轴
    # 设置刻度间隔
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
def draw_map_cir(obstacles, params,ax):
    # Bounds on world
    world_bounds_x =[10,140] #[params.world_bounds_x[0], params.world_bounds_x[1]]
    world_bounds_y = [30,185]#[params.world_bounds_y[0], params.world_bounds_y[1]]
    # Draw obstacles
    #ax = plt.gca()
    ax.set_xlim(world_bounds_x)
    ax.set_ylim(world_bounds_y)
    ax.set_aspect('equal')
    for k in range(len(obstacles)):
        x_center, y_center, radius = obstacles[k]
        circle = Circle((x_center, y_center), radius, color='#777777')
        ax.add_patch(circle)
    plt.axis('on')  # 启用坐标轴
    # 设置刻度间隔
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
    ax.yaxis.set_major_locator(plt.MultipleLocator(50))  # 每 20 米一个刻度
# 读取障碍物信息
obstacles = []
with open(r'./result/200_hig_obstacles_rec.txt', 'r') as file:
    for line in file:
        obstacles.append(eval(line.strip()))  # 使用eval将字符串转换为列表
obstacles=obstacles[109]
# 读取所有机器人的轨迹信息
import json
import numpy as np
with open(r'./result/fail_compare_trajectory_rec.json', 'r') as file:
    trajectory = json.load(file)
with open(r'./result/fail_compare_compare_trajectory_rec.json', 'r') as file:
    compare_trajectory = json.load(file)
with open(r'./result/fail_compare_xiaorong_trajectory_rec.json', 'r') as file:
    xiaorong_trajectory = json.load(file)
# 手动将字符串键转换为整数
trajectory = {int(k): np.array(v) for k, v in trajectory.items()}  # 再次将列表转换为 NumPy 数组

params = Params()

#colors = ['red', 'green', 'skyblue', 'blue', 'magenta', 'yellow', 'gray', 'purple', 'olive', 'teal', 'pink', 'palegreen', 'coral', 'turquoise', 'navy']
#colors = ['#e84445', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2', '#1999b2']
colors = ['#ed2225', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef', '#00adef']
UAVs = ['UAV$_{1}$', 'UAV$_{2}$', 'UAV$_{3}$', 'UAV$_{4}$', 'UAV$_{5}$', 'UAV$_{6}$', 'UAV$_{7}$', 'UAV$_{8}$', 'UAV$_{9}$', 'UAV$_{10}$', 'UAV$_{11}$', 'UAV$_{12}$', 'UAV$_{13}$', 'UAV$_{14}$', 'UAV$_{15}$']


# 创建图形和子图
fig, axs = plt.subplots(2, 2, figsize=(6, 8) )

# 绘制第一个子图


#初始时刻
t=0
for i in range(params.num_robots):
    axs[0,0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
R=11.5
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0,0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[0,0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[0,0].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
axs[0,0].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0,0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

#中间时刻2
t=int(len(trajectory[0])/4)+200
for i in range(params.num_robots):
    axs[0,0].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0,0].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#'''
axs[0,0].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#777777', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[0,0].legend(
    loc='upper left',      # 设置图例位置为右上角
    bbox_to_anchor=(-0.02, 1.02),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.05 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)#'''
draw_map(obstacles, params,axs[0,0])
#axs[0,0].set_title('BPF-FOA', fontsize=16)
#axs[0].set_xlabel('x-axis', fontsize=16)
#axs[0].set_ylabel('y-axis', fontsize=16)
axs[0,0].tick_params(axis='both', which='major', labelsize=16)
axs[0,0].set_xticks([])  # 取消x轴标记
axs[0,0].set_yticks([])  # 取消y轴标记

axs[0,1].plot(trajectory[0][-1][0], trajectory[0][-1][1], '*', color=colors[0],label='Target', markersize=6, zorder=10)
axs[1,1].plot(trajectory[0][-1][0], trajectory[0][-1][1], '*', color=colors[0],label='Target', markersize=6, zorder=10)
# 绘制第二个子图
draw_map(obstacles, params,axs[0,1])

trajectory = {int(k): np.array(v) for k, v in compare_trajectory.items()}  # 再次将列表转换为 NumPy 数组
#初始时刻
t=0
for i in range(params.num_robots):
    axs[0,1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0,1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[0,1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[0,1].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
#axs[1].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)


# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[0,1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线

#axs[1].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#4D4D4D', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[0,1].legend(
    loc='lower left',      # 设置图例位置为右上角
    bbox_to_anchor=(-0.02, -0.02),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
#axs[0,1].set_title('CPF-OA', fontsize=16)
#axs[1].set_xlabel('x-axis', fontsize=16)
axs[0,1].tick_params(axis='both', which='major', labelsize=16)
axs[0,1].set_xticks([])  # 取消x轴标记
axs[0,1].set_yticks([])  # 取消y轴标记


# 绘制第三个子图
draw_map(obstacles, params,axs[1,1])
trajectory = {int(k): np.array(v) for k, v in xiaorong_trajectory.items()}  # 再次将列表转换为 NumPy 数组
#初始时刻
t=0
for i in range(params.num_robots):
    axs[1,1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1,1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    axs[1,1].plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    axs[1,1].plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
#axs[2].plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            axs[1,1].plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线


#axs[2].plot((obstacles[-1][0][0] + obstacles[-1][1][0]) / 2, (obstacles[-1][1][1] + obstacles[-1][2][1]) / 2, 's',color='#4D4D4D', markersize=4, label='Obstacles')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=6, fontsize=10, columnspacing=0.5,handlelength=1.5)
axs[1,1].legend(
    loc='lower left',  # 设置图例位置为右上角
    bbox_to_anchor=(-0.02, -0.02),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
#axs[1,1].set_title('BPF-OA', fontsize=16)
#axs[2].set_xlabel('x-axis', fontsize=16)
axs[1,1].tick_params(axis='both', which='major', labelsize=16)
axs[1,1].set_xticks([])  # 取消x轴标记
axs[1,1].set_yticks([])  # 取消y轴标记

ax3=axs[1,0]
with open(r'./result/fail_compare_compare_trajectory_cir.json', 'r') as file:
    compare_trajectory = json.load(file)

# 绘制第4个子图
obstacles = []
with open(r'./result/200_hig_obstacles_cir.txt', 'r') as file:
    for line in file:
        obstacles.append(eval(line.strip()))  # 使用eval将字符串转换为列表
obstacles=obstacles[9]
draw_map_cir(obstacles, params,ax3)
trajectory = {int(k): np.array(v) for k, v in compare_trajectory.items()}  # 再次将列表转换为 NumPy 数组
#初始时刻
t=0
for i in range(params.num_robots):
    ax3.plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            ax3.plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
#最后时刻
t=-1
for i in range(params.num_robots):
    ax3.plot(trajectory[i][t][0], trajectory[i][t][1], 's', color=colors[i], markersize=4, zorder=3)
    ax3.plot(trajectory[i][:, 0], trajectory[i][:, 1], linewidth=0.5, color=colors[i],markersize=10, zorder=1,alpha=0.7)
#ax3.plot(trajectory[0][t][0], trajectory[0][t][1], 's', color=colors[0], label=UAVs[0], markersize=4, zorder=10)
# 检查每对点之间的距离，并在距离小于 R 时画线
for i in range(params.num_robots):
    for j in range(i + 1, params.num_robots):
        distance = np.linalg.norm(trajectory[i][t] -trajectory[j][t] )  # 计算两点之间的欧几里得距离
        if distance < R:
            ax3.plot([trajectory[i][t][0], trajectory[j][t][0]], [trajectory[i][t][1],trajectory[j][t][1]], 'k-',linewidth=1,zorder=2)  # 画线
        if distance<1.5:
            ax3.plot(trajectory[i][t][0], trajectory[i][t][1], 's', color='#4f8b57', markersize=4, zorder=3)
            ax3.plot(trajectory[j][t][0], trajectory[j][t][1], 's', color='#4f8b57', markersize=4, zorder=3)

ax3.plot(obstacles[-1][0], obstacles[-1][1], 'o', color='#777777', markersize=4, label='Obstacles')
ax3.legend(
    loc='lower left',      # 设置图例位置为右上角
    bbox_to_anchor=(-0.02,-0.02),  # 锚点位置设置为右上角
    ncol=1,                  # 图例分为 6 列
    fontsize=16,             # 字体大小为 10
    columnspacing=0.1,       # 列间距为 0.5
    handlelength=0.5 ,         # 图例条目长度为 1.5
    markerscale=2,
labelspacing=0.1,
)
#ax3.set_title('CPF-OA', fontsize=16)
#axs[1].set_xlabel('x-axis', fontsize=16)
ax3.tick_params(axis='both', which='major', labelsize=16)
ax3.set_xticks([])  # 取消x轴标记
ax3.set_yticks([])  # 取消y轴标记

# 在第一个子图中添加(a)标识
axs[0,0].text(0.5 ,-0.05, "(a)", fontsize=16, ha='center', va='center', transform=axs[0,0].transAxes)

# 在第二个子图中添加(b)标识
axs[0,1].text(0.5 ,-0.05, "(b)", fontsize=16, ha='center', va='center', transform=axs[0,1].transAxes)

# 在第三个子图中添加(c)标识
axs[1,0].text(0.5,-0.05, "(c)", fontsize=16, ha='center', va='center', transform=axs[1,0].transAxes)
# 在第三个子图中添加(c)标识
axs[1,1].text(0.5,-0.05, "(d)", fontsize=16, ha='center', va='center', transform=axs[1,1].transAxes)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.02 )
#plt.subplots_adjust(hspace=0.001 )
plt.savefig('fig_four_simulation_results.jpg', dpi=600)  # 设置保存的分辨率

plt.show()