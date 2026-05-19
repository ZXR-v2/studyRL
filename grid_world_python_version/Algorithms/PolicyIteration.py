"""Policy Iteration for the GridWorld environment.

这个文件实现表格型策略迭代算法，包括两种策略评估模式：

1. 非 truncated 模式：每次 policy evaluation 都一直迭代到收敛。
2. truncated 模式：每次 policy evaluation 只做固定轮数，然后立刻进入 policy improvement。

两种模式的区别只在 policy evaluation 的“评估深度”，policy improvement 的逻辑相同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolicyIterationResult:
    """保存策略迭代的最终结果和中间历史。"""

    values: np.ndarray
    policy: np.ndarray
    q_values: np.ndarray
    policy_iterations: int
    total_evaluation_iterations: int
    stable: bool
    history: list[dict]


class PolicyIteration:
    """对确定性 GridWorld 做策略迭代。

    策略迭代由两步交替组成：

    1. Policy Evaluation:
       固定当前策略 pi，计算该策略对应的状态价值 V_pi(s)。

    2. Policy Improvement:
       根据当前 V_pi(s) 计算每个动作的 Q(s, a)，然后把策略改成贪心策略。

    如果策略改进前后不再变化，说明策略已经稳定，算法结束。
    """

    def __init__(
        self,
        env,
        gamma=0.9,
        theta=1e-6,
        max_policy_iterations=100,
        max_evaluation_iterations=1000,
        truncated=False,
        truncated_evaluation_iterations=1,
    ):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_policy_iterations = max_policy_iterations
        self.max_evaluation_iterations = max_evaluation_iterations
        self.truncated = truncated
        self.truncated_evaluation_iterations = truncated_evaluation_iterations

        self.num_states = env.num_states
        self.num_actions = len(env.action_space)

        # values[i] 表示第 i 个状态的状态价值 V(s)。
        self.values = np.zeros(self.num_states)

        # q_values[i, j] 表示第 i 个状态执行第 j 个动作的动作价值 Q(s, a)。
        self.q_values = np.zeros((self.num_states, self.num_actions))

        # policy[i, j] 表示第 i 个状态选择第 j 个动作的概率。
        self.policy = self._initial_policy()

        # history 保存每次“策略评估 + 策略改进”后的结果，方便打印中间过程。
        self.history = []

    def _initial_policy(self):
        """初始化策略。

        普通状态使用均匀随机策略：每个动作概率都是 1 / 动作数。
        目标状态是终止状态，到达后不再选择动作，所以该行保持全 0。
        """
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions

        target_index = self.state_to_index(self.env.target_state)
        policy[target_index, :] = 0.0
        return policy

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
        """运行策略迭代。

        Args:
            verbose: True 时打印每轮策略迭代的中间结果。
            precision: 打印价值函数时保留的小数位数。
        """
        self.history = []
        total_evaluation_iterations = 0
        stable = False

        for policy_iteration in range(1, self.max_policy_iterations + 1):
            evaluation_iterations, delta = self.policy_evaluation()
            total_evaluation_iterations += evaluation_iterations

            stable = self.policy_improvement()

            self.history.append(
                {
                    "policy_iteration": policy_iteration,
                    "evaluation_iterations": evaluation_iterations,
                    "delta": delta,
                    "stable": stable,
                    "values": self.values.copy(),
                    "policy": self.policy.copy(),
                }
            )

            if verbose:
                self.print_iteration(
                    policy_iteration=policy_iteration,
                    evaluation_iterations=evaluation_iterations,
                    delta=delta,
                    stable=stable,
                    precision=precision,
                )

            if stable:
                break

        return PolicyIterationResult(
            values=self.values.copy(),
            policy=self.policy.copy(),
            q_values=self.q_values.copy(),
            policy_iterations=policy_iteration,
            total_evaluation_iterations=total_evaluation_iterations,
            stable=stable,
            history=self.history.copy(),
        )

    def policy_evaluation(self):
        """在当前策略下评估 V_pi(s)。

        非 truncated 模式：
            一直迭代，直到所有状态价值的最大变化量 delta 小于 theta，
            或者达到 max_evaluation_iterations。

        truncated 模式：
            只做 truncated_evaluation_iterations 轮评估。
            这时 V_pi(s) 不一定完全收敛，但可以更快进入策略改进。
        """
        if self.truncated:
            evaluation_limit = self.truncated_evaluation_iterations
        else:
            evaluation_limit = self.max_evaluation_iterations

        last_delta = float("inf")

        for evaluation_iteration in range(1, evaluation_limit + 1):
            old_values = self.values.copy()
            last_delta = 0.0

            for state_index in range(self.num_states):
                state = self.index_to_state(state_index)

                # 目标状态是终止状态。到达目标后 episode 结束，因此 V(target)=0。
                if state == self.env.target_state:
                    self.values[state_index] = 0.0
                    continue

                # Bellman expectation equation:
                # V_pi(s) = sum_a pi(a|s) * [R(s,a) + gamma * V_pi(s')]
                #
                # 这里先把当前状态的所有动作都枚举出来，
                # 再按当前策略 policy[state_index, action_index] 做加权平均。
                new_value = 0.0
                for action_index, action in enumerate(self.env.action_space):
                    action_probability = self.policy[state_index, action_index]

                    # 如果某个动作概率为 0，它对期望价值没有贡献，可以跳过。
                    if action_probability == 0:
                        continue

                    next_state, reward = self.env._get_next_state_and_reward(state, action)
                    next_state_index = self.state_to_index(next_state)
                    one_step_return = reward + self.gamma * old_values[next_state_index]
                    new_value += action_probability * one_step_return

                self.values[state_index] = new_value
                last_delta = max(last_delta, abs(new_value - old_values[state_index]))

            # 非 truncated 模式需要检查是否收敛。
            # truncated 模式故意不提前用 theta 停止，而是固定评估轮数后就去改进策略。
            if not self.truncated and last_delta < self.theta:
                break

        return evaluation_iteration, last_delta

    def policy_improvement(self):
        """根据当前 values 改进策略。

        对每个状态，先计算每个动作的 Q(s,a)：

            Q(s,a) = R(s,a) + gamma * V(s')

        然后选择 Q 最大的动作作为新的贪心策略。
        如果多个动作并列最优，就把概率平均分给这些动作。

        Returns:
            True 表示策略已经稳定，没有发生变化；False 表示策略仍被更新。
        """
        old_policy = self.policy.copy()
        self.policy.fill(0.0)

        for state_index in range(self.num_states):
            state = self.index_to_state(state_index)

            # 终止状态不需要选择动作，策略保持全 0，动作价值也保持全 0。
            if state == self.env.target_state:
                self.q_values[state_index, :] = 0.0
                continue

            action_values = []
            for action in self.env.action_space:
                next_state, reward = self.env._get_next_state_and_reward(state, action)
                next_state_index = self.state_to_index(next_state)
                action_value = reward + self.gamma * self.values[next_state_index]
                action_values.append(action_value)

            self.q_values[state_index, :] = action_values

            # np.isclose 用来处理浮点数误差，避免 1.0000000001 和 1.0 被错误看成不同。
            # flatnonzero 会返回所有 True 的位置，也就是所有并列最优动作编号。
            best_value = np.max(self.q_values[state_index])
            best_actions = np.flatnonzero(np.isclose(self.q_values[state_index], best_value))

            # 如果有多个最优动作，就平均分配概率。
            # 例如最优动作是 [1, 2]，策略就是 [0, 0.5, 0.5, 0, 0]。
            self.policy[state_index, best_actions] = 1.0 / len(best_actions)

        return np.allclose(old_policy, self.policy)

    def values_as_grid(self):
        """把一维价值函数整理成二维网格，方便打印观察。"""
        return self.values.reshape((self.env.env_size[1], self.env.env_size[0]))

    def print_iteration(self, policy_iteration, evaluation_iterations, delta, stable, precision=2):
        """打印某次策略迭代后的价值函数和策略状态。"""
        mode = "truncated" if self.truncated else "full"
        print(
            f"\nPolicy iteration {policy_iteration} "
            f"({mode} evaluation, {evaluation_iterations} eval sweeps), "
            f"delta = {delta:.8f}, stable = {stable}"
        )
        print(np.round(self.values_as_grid(), precision))
