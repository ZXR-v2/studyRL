
import sys
from pathlib import Path

# 允许示例脚本从项目根目录或 examples 目录启动时都能导入 src.grid_world。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.grid_world import GridWorld
from examples.arguments import args
import random
import numpy as np

# 示例程序：随机选择动作与 GridWorld 交互，并在最后叠加随机策略和值函数。
if __name__ == "__main__":             
    env = GridWorld()

    # reset 返回初始状态和额外信息；这里主要用于初始化环境和轨迹。
    state = env.reset()               

    # 随机策略交互 max_steps 步：每步先渲染，再随机采样动作并执行。
    for t in range(args.max_steps):
        env.render()
        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)

        # 程序内部坐标是 0-based：第一行/第一列编号为 0，例如 (0, 0)。
        # 打印时加上 [1, 1]，转换成 1-based：第一行/第一列编号为 1，例如 (1, 1)。
        # 这样输出更接近日常读表格时的“第几行、第几列”。
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

        # 如果希望到达目标后立即停止，可以取消下面两行注释。
        # if done:
        #     break
    
    # 随机生成一个策略矩阵。默认 5x5 网格有 25 个状态，动作空间有 5 个动作，
    # 所以 policy_matrix 的形状是 (25, 5)：每一行是一个状态，每一列是一个动作。
    # 每个元素表示“在这个状态下选择这个动作的概率”。
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            

    # 归一化每一行，让每个状态下所有动作概率之和等于 1。
    # axis=1 表示按行求和；[:, np.newaxis] 把行和变成列向量，方便逐行相除。
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    # 在当前图上绘制策略箭头/圆圈。
    env.add_policy(policy_matrix)

    
    # 随机生成状态价值，并显示到对应网格中心。
    values = np.random.uniform(0,10,(env.num_states,))
    env.add_state_values(values)

    # 最后再渲染一次，暂停 2 秒以便观察策略和值函数标注。
    env.render(animation_interval=2)
