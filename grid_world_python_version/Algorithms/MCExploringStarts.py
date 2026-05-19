"""Monte Carlo control with exploring starts for GridWorld.

这个文件和 MCBasic.py 的核心区别：

1. MCBasic 是“按轮次”的 model-free policy iteration：
   - 第 k 轮对每个 (s,a) 收集多条 episode。
   - 用这些 episode 的平均 return 得到 q_k(s,a)。
   - 完成整轮 evaluation 后统一改进策略。

2. MCExploringStarts 是 control：
   - 每个 episode 随机选择起始状态和起始动作，这就是 exploring starts。
   - 估计动作价值 Q(s, a)，而不是只估计 V(s)。
   - 每个 episode 内从后往前更新 Returns(s,a)、Num(s,a)、q(s,a)，并改进访问到的状态策略。

Exploring starts 的目的，是保证每个合法的 state-action pair 都有机会被采样到。
如果只从固定起点出发，很多远处状态或某些动作可能很少被访问，Q(s,a) 会很难学准。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCExploringStartsResult:
    """保存 Monte Carlo Exploring Starts 的输出结果。

    Attributes:
        values: 一维数组，values[s] = max_a q(s,a)，用于显示状态价值。
        q_values: 二维数组，q_values[s,a] 是动作价值估计。
        policy: 二维数组，policy[s,a] 是最终策略选择动作 a 的概率。
        visit_counts: 二维数组，记录每个 (s,a) 被 every-visit/first-visit 更新的次数。
        episodes: 总共采样的 episode 数量。
        history: 每个 episode 后的快照，方便调试或画曲线。
    """

    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    visit_counts: np.ndarray
    episodes: int
    history: list[dict]


class MCExploringStarts:
    """使用 exploring starts 的 Monte Carlo control 算法。

    这个算法学习的是最优动作价值 Q(s,a)，并从 Q(s,a) 推出贪心策略。
    它和 MCBasic 最大的不同是：

    - MCBasic 每轮系统性遍历所有 (s,a)，这里每个 episode 随机抽一个 (s0,a0)。
    - MCBasic 每轮结束后统一改进策略，这里在每个 episode 中边更新 Q 边改进访问到的状态。
    """

    def __init__(
        self,
        env,
        gamma=0.9,
        num_episodes=1000,
        max_steps_per_episode=100,
        first_visit=False,
        seed=None,
        include_forbidden_starts=True,
    ):
        """初始化 MC Exploring Starts 算法。

        Args:
            env: GridWorld 环境实例。
            gamma: 折扣因子，控制未来奖励在回报 G 中的权重。
            num_episodes: 总共采样多少条 episode。
            max_steps_per_episode: 每条 episode 的最大步数，防止无法到达终点时无限循环。
            first_visit: False 表示 every-visit MC，和 PPT 每个 step 都更新一致；
                True 表示 first-visit 变体，同一 episode 中同一个 (s,a) 只更新一次。
            seed: 随机数种子；传入整数可复现实验。
            include_forbidden_starts: exploring starts 是否允许从 forbidden state 开始。
                PPT 要求所有 state-action pair 都可能被选中，所以默认 True。
        """
        self.env = env
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        # PPT 的伪代码对 episode 中每个 step 都执行更新，没有跳过重复出现的 (s,a)，
        # 因此默认 first_visit=False，即 every-visit MC。
        # 如果想实验 first-visit 变体，可以手动传 first_visit=True。
        self.first_visit = first_visit
        self.include_forbidden_starts = include_forbidden_starts
        self.rng = np.random.default_rng(seed)

        self.num_states = env.num_states
        self.num_actions = len(env.action_space)

        # q_values[i, j] 表示状态 i 下执行动作 j 的价值 Q(s,a)。
        # MC Exploring Starts 学的是 Q(s,a)，因为 control 需要知道“哪个动作更好”。
        self.q_values = np.zeros((self.num_states, self.num_actions))

        # values[i] 只是为了可视化，把每个状态的 max_a Q(s,a) 当作 V(s) 展示。
        self.values = np.zeros(self.num_states)

        # returns_sum[i, j] 累加 state-action pair (s,a) 观察到的所有回报 G。
        self.returns_sum = np.zeros((self.num_states, self.num_actions))

        # visit_counts[i, j] 记录 (s,a) 被用于 MC 更新的次数。
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=int)

        # policy[i, j] 是当前策略 pi(a|s)。初始化为均匀随机策略。
        self.policy = self._uniform_random_policy()

        # exploring_start_pairs 保存所有可作为 episode 开头的 (state, action_index)。
        self.exploring_start_pairs = self._build_exploring_start_pairs()

        # history 保存每个 episode 后的摘要，方便观察学习过程。
        self.history = []

    def _uniform_random_policy(self):
        """创建均匀随机策略。

        目标状态是终止状态，不需要动作，因此目标状态那一行保持全 0。

        Returns:
            形状为 (num_states, num_actions) 的初始策略矩阵。
        """
        policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        target_index = self.state_to_index(self.env.target_state)
        policy[target_index, :] = 0.0
        return policy

    def _build_exploring_start_pairs(self):
        """列出所有允许作为 exploring start 的 state-action pair。

        MCBasic 的起点来自 env.reset()，也就是固定的 start_state。

        这里不同：我们提前列出所有合法起始状态，再和每个动作组合成
        (state, action) 对。每个 episode 从这些 pair 中随机选一个开始。

        PPT 要求 all pairs can be possibly selected，所以默认包括 forbidden_states。
        目标状态是终止状态，不作为 episode 起点。

        Returns:
            列表，每个元素是 (state, action_index)，表示一个可被随机抽到的起始对。
        """
        pairs = []
        forbidden_states = set(self.env.forbidden_states)

        for state_index in range(self.num_states):
            state = self.index_to_state(state_index)

            if state == self.env.target_state:
                continue
            if not self.include_forbidden_starts and state in forbidden_states:
                continue

            for action_index in range(self.num_actions):
                pairs.append((state, action_index))

        return pairs

    def state_to_index(self, state):
        """把 (x, y) 状态坐标转换成一维状态编号。

        Args:
            state: 二维网格坐标，例如 (2, 3)。

        Returns:
            一维状态编号。
        """
        x, y = state
        return y * self.env.env_size[0] + x

    def index_to_state(self, index):
        """把一维状态编号转换成 (x, y) 状态坐标。

        Args:
            index: 一维状态编号。

        Returns:
            对应的二维状态坐标 (x, y)。
        """
        x = index % self.env.env_size[0]
        y = index // self.env.env_size[0]
        return (x, y)

    def choose_action_index(self, state):
        """按照当前策略 pi(a|s) 采样动作编号。

        Args:
            state: 当前状态坐标。

        Returns:
            动作编号 action_index。真正的动作元组是 env.action_space[action_index]。
        """
        state_index = self.state_to_index(state)
        action_probabilities = self.policy[state_index]
        return self.rng.choice(self.num_actions, p=action_probabilities)

    def choose_exploring_start(self):
        """随机选择 episode 的起始 state-action pair。

        这是和 MCBasic 最关键的差异：

        - MCBasic: state, _ = env.reset()，起点固定。
        - Exploring Starts: 随机选择 (state, first_action)，起点和第一步动作都随机。

        Returns:
            (start_state, first_action_index)，作为新 episode 的起始 state-action pair。
        """
        pair_index = self.rng.integers(len(self.exploring_start_pairs))
        return self.exploring_start_pairs[pair_index]

    def generate_episode(self):
        """用 exploring starts 和当前策略采样一条 episode。

        episode 中每个元素是 (state, action_index, reward)。

        第一步动作来自 exploring starts 随机指定；
        第二步开始，动作才来自当前策略 self.policy。

        Returns:
            episode 列表，其中每个元素是 (state, action_index, reward)。
        """
        start_state, first_action_index = self.choose_exploring_start()

        # GridWorld.reset() 只能回到默认 start_state。
        # Exploring starts 要从随机状态开始，所以这里手动设置 agent_state 和轨迹。
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

            # 只有第一步动作由 exploring starts 指定。
            # 后续动作全部按当前策略 pi(a|s) 采样。
            action_index = self.choose_action_index(state)

        return episode

    def run(self, verbose=False, print_every=100, precision=2):
        """运行 Monte Carlo Exploring Starts control。

        Args:
            verbose: True 时定期打印当前价值估计和策略。
            print_every: verbose=True 时，每隔多少个 episode 打印一次。
            precision: 打印价值函数时保留的小数位数。

        Returns:
            MCExploringStartsResult，包含最终 q_values、policy、visit_counts 和历史记录。
        """
        self.history = []

        for episode_index in range(1, self.num_episodes + 1):
            episode = self.generate_episode()

            # G 是从当前时间步开始到 episode 结束的折扣回报。
            # 从后往前扫，可以用 G = reward + gamma * G 递推。
            G = 0.0

            # PPT 中写的是 For each step of the episode 都更新：
            # Returns(s_t,a_t)、Num(s_t,a_t)、q(s_t,a_t)。
            # 所以默认 first_visit=False，也就是 every-visit。
            # 若 first_visit=True，则变成 first-visit 变体：同一 episode 内同一个
            # (s,a) 只在第一次访问处更新。
            visited_state_action_pairs = set()

            for state, action_index, reward in reversed(episode):
                state_index = self.state_to_index(state)
                G = reward + self.gamma * G

                pair = (state_index, action_index)
                if self.first_visit and pair in visited_state_action_pairs:
                    continue
                visited_state_action_pairs.add(pair)

                # 样本平均更新 Q(s,a)：
                # Q(s,a) = average(所有从 (s,a) 开始观察到的回报 G)
                self.returns_sum[state_index, action_index] += G
                self.visit_counts[state_index, action_index] += 1
                self.q_values[state_index, action_index] = (
                    self.returns_sum[state_index, action_index]
                    / self.visit_counts[state_index, action_index]
                )

                # PPT 的 policy improvement 写在每个 step 内：
                # pi(a|s_t)=1 if a=argmax_a q(s_t,a)。
                # 因此这里只改进当前刚更新过的 state，而不是每次重扫所有状态。
                self.improve_policy_for_state(state_index)

            self.update_state_values_from_q()

            reached_target = bool(episode and self.env._is_done(self.env.agent_state))
            self.history.append(
                {
                    "episode": episode_index,
                    "episode_length": len(episode),
                    "reached_target": reached_target,
                    "values": self.values.copy(),
                    "q_values": self.q_values.copy(),
                    "policy": self.policy.copy(),
                    "visit_counts": self.visit_counts.copy(),
                }
            )

            if verbose and (episode_index == 1 or episode_index % print_every == 0):
                self.print_episode_summary(episode_index, episode, reached_target, precision)

        return MCExploringStartsResult(
            values=self.values.copy(),
            q_values=self.q_values.copy(),
            policy=self.policy.copy(),
            visit_counts=self.visit_counts.copy(),
            episodes=self.num_episodes,
            history=self.history.copy(),
        )

    def improve_policy_for_state(self, state_index):
        """根据当前 Q(s,a) 改进某一个状态的策略。

        PPT 写的是：

            pi(a|s_t) = 1 if a = argmax_a q(s_t,a), otherwise 0

        因此这里使用确定性 argmax。若出现并列最大值，np.argmax 会选择第一个。

        Args:
            state_index: 要被改进策略的一维状态编号。

        Returns:
            None。该方法直接原地修改 self.policy[state_index]。
        """
        state = self.index_to_state(state_index)
        if state == self.env.target_state:
            return

        self.policy[state_index, :] = 0.0
        best_action = int(np.argmax(self.q_values[state_index]))
        self.policy[state_index, best_action] = 1.0

    def improve_policy(self):
        """把所有非终止状态的策略都改成当前 Q 下的贪心策略。

        Returns:
            None。该方法直接原地修改 self.policy。
        """
        for state_index in range(self.num_states):
            self.improve_policy_for_state(state_index)

    def update_state_values_from_q(self):
        """用 max_a Q(s,a) 得到用于显示的状态价值 V(s)。

        Returns:
            None。该方法直接原地更新 self.values。
        """
        self.values = np.max(self.q_values, axis=1)
        target_index = self.state_to_index(self.env.target_state)
        self.values[target_index] = 0.0

    def values_as_grid(self):
        """把一维状态价值整理成二维网格。

        Returns:
            形状为 (rows, cols) 的二维数组。
        """
        return self.values.reshape((self.env.env_size[1], self.env.env_size[0]))

    def visit_counts_as_grid(self):
        """把 state-action 访问次数按状态求和后整理成二维网格。

        Returns:
            每个状态累计访问次数的二维网格。
        """
        state_visit_counts = np.sum(self.visit_counts, axis=1)
        return state_visit_counts.reshape((self.env.env_size[1], self.env.env_size[0]))

    def print_episode_summary(self, episode_index, episode, reached_target, precision=2):
        """打印某个 episode 后的学习状态。

        Args:
            episode_index: 当前 episode 编号。
            episode: 当前 episode 的轨迹列表。
            reached_target: 当前 episode 是否到达目标状态。
            precision: 打印状态价值时保留的小数位数。
        """
        print(
            f"\nEpisode {episode_index}: "
            f"length = {len(episode)}, reached_target = {reached_target}"
        )
        print("State values from max_a Q(s,a):")
        print(np.round(self.values_as_grid(), precision))
        print("State-action visit counts summed by state:")
        print(self.visit_counts_as_grid())
