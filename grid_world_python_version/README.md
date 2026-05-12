# Grid World Python Version

这个目录实现了一个简单的二维 Grid World 环境，适合用于强化学习作业中的环境交互、随机策略演示、策略可视化和状态价值可视化。

## 目录结构

```text
grid_world_python_version/
├── README.md
├── examples/
│   ├── arguments.py
│   └── example_grid_world.py
├── plots/
│   ├── sample1.png
│   ├── sample2.png
│   ├── sample3.png
│   └── sample4.png
└── src/
    └── grid_world.py
```

## 各目录和文件作用

### `src/`

存放环境的核心代码。

- `grid_world.py`：定义 `GridWorld` 类，是整个项目最重要的文件。
  - `reset()`：重置环境到起点。
  - `step(action)`：执行一步动作，返回下一状态、奖励、是否结束和额外信息。
  - `_get_next_state_and_reward(state, action)`：根据当前状态和动作计算下一状态与奖励。
  - `_is_done(state)`：判断是否到达目标状态。
  - `render()`：使用 Matplotlib 绘制网格、智能体位置和轨迹。
  - `add_policy(policy_matrix)`：把策略矩阵画成箭头或圆圈。
  - `add_state_values(values)`：在每个格子中显示状态价值。

### `examples/`

存放参数配置和示例程序。

- `arguments.py`：定义环境默认参数。
  - 网格大小：`env_size`
  - 起点：`start_state`
  - 目标：`target_state`
  - 禁区：`forbidden_states`
  - 奖励：`reward_target`、`reward_forbidden`、`reward_step`
  - 动作空间：`action_space`
  - 动画间隔：`animation_interval`

- `example_grid_world.py`：演示如何使用 `GridWorld`。
  - 创建环境。
  - 重置环境。
  - 随机选择动作并和环境交互。
  - 绘制随机策略。
  - 绘制随机状态价值。

### `plots/`

存放示例输出图片，例如环境网格、策略箭头或状态价值图。这个目录主要用于展示运行效果。

### `__pycache__/`

Python 自动生成的缓存目录，不属于作业源码，不需要手动修改。

## 环境规则

默认环境参数在 `examples/arguments.py` 中定义：

- 默认网格大小为 `5 x 5`。
- 默认起点为 `(2, 2)`。
- 默认目标为 `(4, 4)`。
- 默认禁区为 `(2, 1)`、`(3, 3)`、`(1, 3)`。
- 到达目标奖励为 `10`。
- 撞墙或进入禁区奖励为 `-5`。
- 普通移动奖励为 `-1`。

状态坐标使用 `(x, y)` 表示：

- `x` 表示列索引。
- `y` 表示行索引。
- 程序内部从 `0` 开始计数。
- 图中显示的坐标标签从 `1` 开始，更方便阅读。

默认动作空间为：

```python
[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
```

含义分别是：

- `(0, 1)`：向下
- `(1, 0)`：向右
- `(0, -1)`：向上
- `(-1, 0)`：向左
- `(0, 0)`：原地不动

## 运行示例

如果已经有 `.venv` 虚拟环境，先激活它：

```bash
source .venv/bin/activate
```

如果没有虚拟环境，可以在当前目录创建并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在 `grid_world_python_version` 目录下运行：

```bash
python examples/example_grid_world.py
```

示例程序会随机选择动作与环境交互，并用 Matplotlib 显示智能体轨迹。运行结束后，会在图上叠加随机生成的策略和状态价值。

如果只想快速检查程序是否能运行，可以减少交互步数：

```bash
python examples/example_grid_world.py --max-steps 5 --animation-interval 0
```

## 基本使用方式

可以在自己的强化学习算法中这样使用环境：

```python
from src.grid_world import GridWorld

env = GridWorld()
state, info = env.reset()

done = False
while not done:
    action = env.action_space[0]
    next_state, reward, done, info = env.step(action)
    state = next_state
```

如果需要显示环境：

```python
env.render()
```

如果已经计算出策略矩阵和状态价值，可以调用：

```python
env.add_policy(policy_matrix)
env.add_state_values(values)
env.render()
```

其中：

- `policy_matrix` 的形状应该是 `(env.num_states, len(env.action_space))`。
- `values` 的长度应该是 `env.num_states`。

## 作业理解重点

这个项目可以看作强化学习中的环境部分。算法不在环境内部实现，而是由外部代码调用环境提供的接口：

1. 调用 `reset()` 得到初始状态。
2. 根据当前状态选择动作。
3. 调用 `step(action)` 得到下一状态和奖励。
4. 根据奖励更新自己的策略或价值函数。
5. 重复以上过程，直到到达目标或达到最大步数。

因此，`GridWorld` 的职责是提供状态转移、奖励反馈和可视化；具体的强化学习算法可以另外写在新的脚本中。
