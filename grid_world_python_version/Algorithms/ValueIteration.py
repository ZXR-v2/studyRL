"""Value Iteration for the GridWorld environment.

这个文件实现表格型价值迭代算法。算法只依赖 GridWorld 暴露出来的网格大小、
动作空间、目标状态和状态转移/奖励规则，因此可以直接把 GridWorld 当作 env 使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValueIterationResult:
    """保存价值迭代的输出结果。"""

    values: np.ndarray
    policy: np.ndarray
    q_values: np.ndarray
    iterations: int
    delta: float
    history: list[dict]


class ValueIteration:
    """对确定性 GridWorld 做价值迭代。

    Bellman 最优方程：

        V(s) = max_a [ R(s, a) + gamma * V(s') ]

    其中 s' 是在状态 s 执行动作 a 后到达的下一状态。
    """

    def __init__(self, env, gamma=0.9, theta=1e-6, max_iterations=1000):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.num_states = env.num_states
        self.num_actions = len(env.action_space)

        # values[i] 表示第 i 个状态的价值 V(s)。
        self.values = np.zeros(self.num_states)

        # q_values[i, j] 表示第 i 个状态执行第 j 个动作的动作价值 Q(s, a)。
        self.q_values = np.zeros((self.num_states, self.num_actions))

        # policy[i, j] 表示第 i 个状态选择第 j 个动作的概率。
        self.policy = np.zeros((self.num_states, self.num_actions))

        # history 保存每一轮迭代结束后的中间结果，方便课后复盘或打印观察。
        self.history = []

    def state_to_index(self, state):
        """把 (x, y) 状态坐标转换成一维状态编号。"""
        x, y = state
        return y * self.env.env_size[0] + x

    def index_to_state(self, index):
        """把一维状态编号转换成 (x, y) 状态坐标。"""
        x = index % self.env.env_size[0]
        y = index // self.env.env_size[0]
        return (x, y)

    def run(self, verbose=False, precision=2):
        """运行价值迭代，返回价值函数、贪心策略和动作价值。

        Args:
            verbose: True 时打印每轮迭代后的价值函数。
            precision: 打印价值函数时保留的小数位数。
        """
        last_delta = float("inf")
        self.history = []

        for iteration in range(1, self.max_iterations + 1):
            old_values = self.values.copy()
            last_delta = 0.0

            for state_index in range(self.num_states):
                state = self.index_to_state(state_index)

                # 目标状态是终止状态。到达目标后 episode 结束，因此这里固定 V(target)=0。
                if state == self.env.target_state:
                    self.values[state_index] = 0.0
                    self.q_values[state_index, :] = 0.0
                    continue

                action_values = []
                for action in self.env.action_space:
                    next_state, reward = self.env._get_next_state_and_reward(state, action)
                    next_state_index = self.state_to_index(next_state)
                    action_value = reward + self.gamma * old_values[next_state_index]
                    action_values.append(action_value)

                self.q_values[state_index, :] = action_values
                self.values[state_index] = np.max(action_values)
                last_delta = max(last_delta, abs(self.values[state_index] - old_values[state_index]))

            self.history.append(
                {
                    "iteration": iteration,
                    "delta": last_delta,
                    "values": self.values.copy(),
                }
            )

            if verbose:
                self.print_iteration(iteration, last_delta, precision=precision)

            if last_delta < self.theta:
                break

        self._build_greedy_policy()
        return ValueIterationResult(
            values=self.values.copy(),
            policy=self.policy.copy(),
            q_values=self.q_values.copy(),
            iterations=iteration,
            delta=last_delta,
            history=self.history.copy(),
        )

    def _build_greedy_policy(self):
        """根据 Q(s, a) 生成贪心策略。

        如果多个动作并列最优，就把概率平均分给这些动作。
        """
        self.policy.fill(0.0)

        for state_index in range(self.num_states):
            state = self.index_to_state(state_index)

            # 终止状态不再选择动作，策略保持全 0。
            if state == self.env.target_state:
                continue

            best_value = np.max(self.q_values[state_index])
            best_actions = np.flatnonzero(np.isclose(self.q_values[state_index], best_value))
            self.policy[state_index, best_actions] = 1.0 / len(best_actions)

    def values_as_grid(self):
        """把一维价值函数整理成二维网格，方便打印观察。"""
        return self.values.reshape((self.env.env_size[1], self.env.env_size[0]))

    def values_to_grid(self, values):
        """把传入的一维价值函数整理成二维网格。"""
        return values.reshape((self.env.env_size[1], self.env.env_size[0]))

    def print_iteration(self, iteration, delta, precision=2):
        """打印某一轮迭代后的价值函数。"""
        print(f"\nIteration {iteration}, delta = {delta:.8f}")
        print(np.round(self.values_as_grid(), precision))

    def best_action_indices_as_grid(self):
        """把每个状态的最优动作编号整理成二维网格，方便打印观察。"""
        best_actions = np.argmax(self.policy, axis=1)
        return best_actions.reshape((self.env.env_size[1], self.env.env_size[0]))
