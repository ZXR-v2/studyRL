__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import sys
from typing import Any

# 允许从项目根目录导入 examples.arguments 中的默认配置。
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from examples.arguments import args

class GridWorld():
    """二维网格世界环境。

    状态使用 (x, y) 坐标表示，其中 x 是列索引，y 是行索引。
    程序内部用 0-based 坐标，也就是第一行/第一列从 0 开始编号；
    图上显示时会转成 1-based 坐标，也就是第一行/第一列从 1 开始编号。
    默认参数来自 examples/arguments.py。该类提供类似强化学习环境的 reset、step
    和 render 接口，便于在示例或算法中交互。
    """

    def __init__(self, env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):

        # 环境结构：网格大小、状态总数、起点、终点和禁区。
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        # 智能体当前状态与强化学习交互参数。
        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        # Matplotlib 画布延迟创建；只有第一次 render 时才初始化。
        self.canvas = None
        self.animation_interval = args.animation_interval

        # 可视化配色。
        self.color_forbid = (0.9290,0.6940,0.125)
        self.color_target = (0.3010,0.7450,0.9330)
        self.color_policy = (0.4660,0.6740,0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0,0,1)



    def reset(self):
        """重置环境到起点，并清空/初始化轨迹记录。"""
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}


    def step(self, action):
        """执行一步动作。

        Args:
            action: 动作元组，必须属于 self.action_space，例如 (1, 0) 表示向右。

        Returns:
            (next_state, reward, done, info): 下一状态、奖励、是否到达终点、额外信息。
        """
        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        # 为轨迹线添加一个中间点，使移动方向在图上更容易看出。
        # 0.03 * randn() 是很小的随机抖动，用来避免来回走时轨迹线完全重叠。
        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()

        # 0.2 * action 是沿动作方向的偏移：向右就往右偏一点，向下就往下偏一点。
        # 这个偏移只影响轨迹显示，不影响真实状态、奖励或终止判断。
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
        
    def _get_next_state_and_reward(self, state, action):
        """根据当前状态和动作计算下一状态与即时奖励。

        规则：
        - 撞到边界：停在边界内，并获得 forbidden 奖励；
        - 到达目标：进入目标格，并获得 target 奖励；
        - 进入禁区：留在原地，并获得 forbidden 奖励；
        - 普通移动：进入新格子，并获得 step 奖励。
        """
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        """判断当前状态是否为终点。"""
        return state == self.target_state
    

    def render(self, animation_interval=args.animation_interval):
        """绘制当前环境状态、智能体位置和历史轨迹。"""
        if self.canvas is None:
            # 第一次渲染时创建图像窗口，并设置网格坐标、刻度和显示方向。
            # self.canvas 是整张图/窗口，可以理解成画纸；self.ax 是画纸上的坐标系。
            plt.ion()                             
            self.canvas, self.ax = plt.subplots()   

            # 坐标范围从 -0.5 到 n-0.5，是为了让整数坐标 (0,0)、(1,0) 等落在格子中心。
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)

            # 刻度放在 -0.5、0.5、1.5 ... 上，这样网格线正好画在每个格子的边界。
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
            self.ax.set_aspect('equal')

            # y 轴反向后，坐标看起来更像矩阵/棋盘：越往下 y 越大。
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                # lb 是内部 0-based 坐标；显示时写成 lb+1，让读者看到 1-based 编号。
                self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')           
            for lb in idx_labels_y:
                # 左侧行号同样从 1 开始显示。
                self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')

            # 默认坐标轴刻度已经被上面的 text 手动替代，所以这里隐藏 Matplotlib 自带刻度。
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)   

            # 绘制目标格和禁区格。
            # 状态坐标表示格子中心，Rectangle 需要左下角坐标，所以 x/y 都减去 0.5。
            self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)     

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            # agent_star 表示智能体当前位置，traj_obj 表示走过的轨迹。
            # ax.plot 返回的是 Line2D 对象列表；左侧的逗号表示取出列表里的唯一一个对象。
            # 这里先传空的 x/y 列表，后面每次 render 再用 set_data 更新位置。
            self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # 每次渲染只更新智能体位置和轨迹数据，避免重复创建图形对象。
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        # zip(*self.traj) 会把 [(x1, y1), (x2, y2), ...] 拆成
        # traj_x = (x1, x2, ...)，traj_y = (y1, y2, ...)。
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        if "agg" in plt.get_backend().lower():
            # Agg 是非交互式后端，适合命令行测试或保存图片；这里不能调用 pause。
            self.canvas.canvas.draw()
        else:
            plt.pause(animation_interval)
        if args.debug:
            input('press Enter to continue...')     


 
    def add_policy(self, policy_matrix):                  
        """在网格上叠加策略箭头。

        policy_matrix 的形状应为 (num_states, len(action_space))，每一行表示某个
        状态下各个动作的概率。非零概率会被画成箭头，原地不动动作会被画成圆圈。
        """
        for state, state_action_group in enumerate(policy_matrix):    
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))
    
    def add_state_values(self, values, precision=1):
        """在每个格子中心显示状态价值。

        Args:
            values: 长度为 num_states 的可迭代对象。
            precision: 显示时保留的小数位数。
        """
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='black')
