"""MC Basic algorithm for GridWorld.

这个实现严格按 PPT 中的 MC Basic algorithm 来写：

1. 初始化策略 pi_0。
2. 第 k 轮迭代时，对每个状态 s 和每个动作 a：
   收集若干条从 (s, a) 开始、后续跟随当前策略 pi_k 的 episode。
3. 用这些 episode 的平均回报估计 q_k(s, a)。
4. 对每个状态做 policy improvement：
   pi_{k+1}(a|s)=1 if a=argmax_a q_k(s,a)，否则为 0。

和之前的 MCBasic prediction 版本不同：

- 之前只估计 V(s)，不改进策略。
- PPT 里的 MC Basic 是 model-free policy iteration，会估计 Q(s,a) 并搜索最优策略。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCBasicResult:
    """保存 MC Basic 的最终结果。

    Attributes:
        values: 一维数组，values[s] = max_a q(s,a)，主要用于可视化状态价值。
        q_values: 二维数组，q_values[s,a] 是 Monte Carlo 估计出的动作价值。
        policy: 二维数组，policy[s,a] 是最终策略在状态 s 选择动作 a 的概率。
        visit_counts: 二维数组，记录最后一轮中每个 (s,a) 采样了多少条 episode。
        policy_iterations: 实际完成了多少轮“评估 + 改进”。
        episodes: 总共采样的 episode 数量。
        stable: 策略最后一轮是否已经不再变化。
        history: 每轮迭代后的快照，便于调试或画学习曲线。
    """

    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    visit_counts: np.ndarray
    policy_iterations: int
    episodes: int
    stable: bool
    history: list[dict]


class MCBasic:
    """PPT 版本的 MC Basic algorithm。

    这是一个 control 算法，不是单纯 prediction。它用 Monte Carlo 采样做
    policy evaluation，再用 argmax Q(s,a) 做 policy improvement。
    """

    def __init__(
        self,
        env,
        gamma=0.9,
        policy_iterations=10,
        episodes_per_pair=20,
        max_steps_per_episode=100,
        seed=None,
    ):
        """初始化 MC Basic 算法。

        Args:
            env: GridWorld 环境实例，算法会使用它的状态空间、动作空间和 step 逻辑。
            gamma: 折扣因子，越接近 1 越重视远期奖励，越接近 0 越重视即时奖励。
            policy_iterations: 外层策略迭代轮数，也就是 PPT 里的 k 最大执行次数。
            episodes_per_pair: 每一轮里，每个 (s,a) 要采样多少条 episode。
                数值越大，q(s,a) 的估计越稳定，但运行越慢。
            max_steps_per_episode: 单条 episode 的最大步数，防止随机策略长时间走不到终点。
            seed: 随机数种子；传入整数可以让实验结果可复现。
        """
        self.env = env
        self.gamma = gamma
        self.policy_iterations = policy_iterations
        self.episodes_per_pair = episodes_per_pair
        self.max_steps_per_episode = max_steps_per_episode
        self.rng = np.random.default_rng(seed)

        self.num_states = env.num_states
        self.num_actions = len(env.action_space)

        # q_values[i, j] 表示状态 i 下动作 j 的 q_k(s,a) 估计。
        self.q_values = np.zeros((self.num_states, self.num_actions))

        # values[i] 用于显示，取 max_a q(s,a)。
        self.values = np.zeros(self.num_states)

        # visit_counts[i, j] 记录当前最后一轮 evaluation 中 (s,a) 采样了多少条 episode。
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=int)

        # 初始策略 pi_0。PPT 只要求 initial guess pi_0，这里用均匀随机策略。
        self.policy = self._uniform_random_policy()

        # 目标状态不需要继续选动作；其余状态都作为 PPT 中的 s in S。
        self.non_terminal_states = [
            self.index_to_state(index)
            for index in range(self.num_states)
            if self.index_to_state(index) != self.env.target_state
        ]

        self.history = []

    def _uniform_random_policy(self):
        """创建初始策略 pi_0。

        Returns:
            形状为 (num_states, num_actions) 的策略矩阵。
            普通状态下每个动作概率相同；目标状态是终止状态，动作概率全为 0。
        """
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        target_index = self.state_to_index(self.env.target_state)
        policy[target_index, :] = 0.0
        return policy

    def state_to_index(self, state):
        """把 (x, y) 状态坐标转换成一维状态编号。

        Args:
            state: 二维网格坐标，例如 (2, 3)。

        Returns:
            一维状态编号，用于索引 values、q_values 和 policy。
        """
        x, y = state
        return y * self.env.env_size[0] + x

    def index_to_state(self, index):
        """把一维状态编号转换成 (x, y) 状态坐标。

        Args:
            index: 一维状态编号。

        Returns:
            对应的二维网格坐标 (x, y)。
        """
        x = index % self.env.env_size[0]
        y = index // self.env.env_size[0]
        return (x, y)

    def choose_action_index(self, state):
        """按照当前策略 pi_k(a|s) 采样动作编号。

        Args:
            state: 当前状态坐标。

        Returns:
            动作编号 action_index，而不是动作元组本身。
            例如 action_index=1 对应 env.action_space[1]。
        """
        state_index = self.state_to_index(state)
        probabilities = self.policy[state_index]
        return self.rng.choice(self.num_actions, p=probabilities)

    def generate_episode_from_pair(self, start_state, first_action_index):
        """收集一条从指定 (s,a) 开始的 episode。

        这正对应 PPT 里的：

            Collect episodes starting from (s,a) following pi_k

        第一动作用传入的 first_action_index；从第二步开始，动作由当前策略 pi_k 采样。

        Args:
            start_state: episode 的起始状态 s。
            first_action_index: 起始动作 a 的编号。

        Returns:
            episode 列表，其中每个元素是 (state, action_index, reward)。
            reward 是执行该 state/action 后收到的奖励。
        """
        self.env.agent_state = start_state
        self.env.traj = [start_state]

        episode = []
        state = start_state
        action_index = first_action_index

        for step_index in range(self.max_steps_per_episode):
            action = self.env.action_space[action_index]
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action_index, reward))

            state = next_state
            if done:
                break

            action_index = self.choose_action_index(state)

        return episode

    def return_from_episode_start(self, episode):
        """计算从 episode 起点开始的折扣回报。

        PPT 的 MC Basic 对 q_k(s,a) 的估计是：

            average return of all episodes starting from (s,a)

        因为每条 episode 都从指定 (s,a) 开始，所以这里只需要计算整条 episode
        从第 0 步开始的回报 G_0。

        Args:
            episode: generate_episode_from_pair 返回的轨迹列表。

        Returns:
            从 episode 第 0 步开始的折扣回报 G_0。
        """
        G = 0.0
        for _state, _action_index, reward in reversed(episode):
            G = reward + self.gamma * G
        return G

    def run(self, verbose=False, precision=2):
        """运行 PPT 版本 MC Basic。

        Args:
            verbose: True 时打印每轮策略迭代后的价值表和采样次数。
            precision: 打印数值时保留的小数位数。

        Returns:
            MCBasicResult，包含最终 values、q_values、policy 和历史记录。
        """
        self.history = []
        total_episodes = 0
        stable = False

        for iteration in range(1, self.policy_iterations + 1):
            old_policy = self.policy.copy()

            # 每一轮 k 都是在当前策略 pi_k 下重新做 policy evaluation。
            # 所以 returns_sum 和 visit_counts 每轮清零，得到的是 q_k(s,a)，
            # 而不是把不同策略下的 return 混在一起平均。
            returns_sum = np.zeros((self.num_states, self.num_actions))
            self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=int)

            for state in self.non_terminal_states:
                state_index = self.state_to_index(state)

                for action_index in range(self.num_actions):
                    returns = []

                    # PPT 写的是 sufficiently many episodes。
                    # 这里用 episodes_per_pair 控制“每个 (s,a) 采样多少条”。
                    for _ in range(self.episodes_per_pair):
                        episode = self.generate_episode_from_pair(state, action_index)
                        G = self.return_from_episode_start(episode)
                        returns.append(G)
                        total_episodes += 1

                    returns_sum[state_index, action_index] = np.sum(returns)
                    self.visit_counts[state_index, action_index] = len(returns)
                    self.q_values[state_index, action_index] = np.mean(returns)

            self.improve_policy()
            self.update_state_values_from_q()
            stable = np.array_equal(old_policy, self.policy)

            self.history.append(
                {
                    "iteration": iteration,
                    "values": self.values.copy(),
                    "q_values": self.q_values.copy(),
                    "policy": self.policy.copy(),
                    "visit_counts": self.visit_counts.copy(),
                    "stable": stable,
                }
            )

            if verbose:
                self.print_iteration(iteration, stable, precision=precision)

            if stable:
                break

        return MCBasicResult(
            values=self.values.copy(),
            q_values=self.q_values.copy(),
            policy=self.policy.copy(),
            visit_counts=self.visit_counts.copy(),
            policy_iterations=iteration,
            episodes=total_episodes,
            stable=stable,
            history=self.history.copy(),
        )

    def improve_policy(self):
        """按 PPT 的 argmax 规则做 policy improvement。

        PPT 写的是：

            pi_{k+1}(a|s) = 1 if a = argmax_a q_k(s,a), otherwise 0

        因此这里选择一个确定性 argmax 动作。若有并列最大值，np.argmax 会取
        第一个最大值对应的动作，这和公式里的单个 argmax 保持一致。

        Returns:
            None。该方法直接原地修改 self.policy。
        """
        self.policy.fill(0.0)

        for state in self.non_terminal_states:
            state_index = self.state_to_index(state)
            best_action = int(np.argmax(self.q_values[state_index]))
            self.policy[state_index, best_action] = 1.0

    def update_state_values_from_q(self):
        """用 max_a Q(s,a) 得到用于显示的状态价值。

        Returns:
            None。该方法直接原地更新 self.values。
        """
        self.values = np.max(self.q_values, axis=1)
        target_index = self.state_to_index(self.env.target_state)
        self.values[target_index] = 0.0

    def values_as_grid(self):
        """把一维状态价值整理成二维网格。

        Returns:
            形状为 (rows, cols) 的二维数组，方便按网格打印。
        """
        return self.values.reshape((self.env.env_size[1], self.env.env_size[0]))

    def visit_counts_as_grid(self):
        """把 state-action 访问次数按状态求和后整理成二维网格。

        Returns:
            每个状态总采样次数的二维网格。
        """
        state_visit_counts = np.sum(self.visit_counts, axis=1)
        return state_visit_counts.reshape((self.env.env_size[1], self.env.env_size[0]))

    def print_iteration(self, iteration, stable, precision=2):
        """打印每轮 MC Basic policy iteration 后的结果。

        Args:
            iteration: 当前第几轮策略迭代。
            stable: 本轮策略改进后策略是否已经稳定。
            precision: 打印状态价值时保留的小数位数。
        """
        print(f"\nMC Basic iteration {iteration}, stable = {stable}")
        print("State values from max_a q(s,a):")
        print(np.round(self.values_as_grid(), precision))
        print("State-action samples summed by state:")
        print(self.visit_counts_as_grid())
