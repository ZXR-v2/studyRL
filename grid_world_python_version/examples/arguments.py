__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
"""Grid World 环境的命令行参数与默认配置。"""

from typing import Union
import numpy as np
import argparse

# 创建参数解析器。其他模块导入本文件时，会直接读取 parse_args 后的 args。
parser = argparse.ArgumentParser("Grid World Environment")

# ==================== User settings ====================
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(5,5) )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(2,2))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4,4))

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[ (2, 1), (3, 3), (1, 3)] )

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 10)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -5)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)
# ==================== End of User settings ====================


# ==================== Advanced Settings ====================
# action_space 中的动作依次表示：下、右、上、左、原地不动。
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay           

# debug=True 时，render 每一步都会等待用户按 Enter，方便逐步观察。
parser.add_argument("--debug", type=bool, default=False)

# 每次渲染后的暂停时间，数值越小动画越快。
parser.add_argument("--animation-interval", type=float, default = 0.2)

# 示例程序最多交互多少步；默认保持原示例的 1000 步。
parser.add_argument("--max-steps", type=int, default=1000)
# ==================== End of Advanced settings ====================


# 使用 parse_known_args 是为了允许其他脚本添加自己的命令行参数。
# 例如 examples/main.py 会添加 --verbose、--no-render 等算法参数；
# GridWorld 只读取这里认识的环境参数，未知参数交给调用脚本继续处理。
args, _ = parser.parse_known_args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    """检查网格大小、起点、终点和禁区是否在合法范围内。"""
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    # 对 x、y 两个维度分别检查，确保所有状态坐标没有超出网格边界。
    for i in range(2):
        assert start_state[i] < env_size[i]
        assert target_state[i] < env_size[i]
        for j in range(len(forbidden_states)):
            assert forbidden_states[j][i] < env_size[i]

# 导入或运行本模块时立即校验默认/命令行参数，尽早发现配置错误。
try:
    validate_environment_parameters(args.env_size, args.start_state, args.target_state, args.forbidden_states)
except ValueError as e:
    print("Error:", e)
