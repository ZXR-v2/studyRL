"""Run Monte Carlo Basic on GridWorld."""

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从项目根目录运行：python examples/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Value Iteration 版本保留在这里，方便之后对照切换。
# from Algorithms.ValueIteration import ValueIteration
# Policy Iteration 版本保留在这里，方便之后对照切换。
# from Algorithms.PolicyIteration import PolicyIteration
from Algorithms.MCBasic import MCBasic
from src.grid_world import GridWorld


ACTION_NAMES = {
    0: "down",
    1: "right",
    2: "up",
    3: "left",
    4: "stay",
}


def print_policy(policy, env):
    """把贪心策略打印成每个格子的动作名称。"""
    best_action_indices = np.argmax(policy, axis=1)
    action_grid = best_action_indices.reshape((env.env_size[1], env.env_size[0]))

    print("\nGreedy policy:")
    for y, row in enumerate(action_grid):
        labels = []
        for x, action in enumerate(row):
            if (x, y) == env.target_state:
                labels.append("target")
            else:
                labels.append(ACTION_NAMES[action])
        print("  " + "  ".join(f"{label:>6}" for label in labels))


def parse_args():
    """解析 Monte Carlo Basic 示例脚本自己的参数。"""
    parser = argparse.ArgumentParser("Run Monte Carlo Basic on GridWorld")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--policy-iterations", type=int, default=10)
    parser.add_argument("--episodes-per-pair", type=int, default=20)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="print intermediate values after every MC Basic iteration")
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--no-render", action="store_true", help="skip Matplotlib visualization")
    return parser.parse_args()


def main():
    args = parse_args()

    env = GridWorld()

    # Value Iteration 调用方式保留作对照；如果之后要切回 Value Iteration，
    # 取消下面这段注释，并注释掉 MCBasic 那段即可。注意还需要在 parse_args 中
    # 恢复 Value Iteration 需要的 theta、max_iterations 等参数。
    #
    # solver = ValueIteration(
    #     env,
    #     gamma=args.gamma,
    #     theta=args.theta,
    #     max_iterations=args.max_policy_iterations,
    # )
    # result = solver.run(verbose=args.verbose, precision=args.precision)

    # Policy Iteration 调用方式保留作对照；如果之后要切回 Policy Iteration，
    # 取消下面这段注释，并注释掉 MCBasic 那段即可。注意还需要在 parse_args 中
    # 恢复 Policy Iteration 需要的 theta、truncated 等参数。
    #
    # solver = PolicyIteration(
    #     env,
    #     gamma=args.gamma,
    #     theta=args.theta,
    #     max_policy_iterations=args.max_policy_iterations,
    #     max_evaluation_iterations=args.max_evaluation_iterations,
    #     truncated=args.truncated,
    #     truncated_evaluation_iterations=args.truncated_evaluation_iterations,
    # )
    # result = solver.run(verbose=args.verbose, precision=args.precision)

    solver = MCBasic(
        env,
        gamma=args.gamma,
        policy_iterations=args.policy_iterations,
        episodes_per_pair=args.episodes_per_pair,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
    )
    result = solver.run(verbose=args.verbose, precision=args.precision)

    print(f"MC Basic finished after {result.policy_iterations} policy iterations.")
    print(f"Collected episodes: {result.episodes}")
    print(f"Policy stable: {result.stable}")

    print("\nState values:")
    print(np.round(solver.values_as_grid(), args.precision))

    print("\nVisit counts:")
    print(solver.visit_counts_as_grid())

    print_policy(result.policy, env)

    if args.no_render:
        return

    # 使用 GridWorld 自带的可视化函数显示 MC 估计出的状态价值。
    # 这里也叠加采样策略 policy；默认是均匀随机策略，所以每个动作都会有箭头/圆圈。
    env.reset()
    env.render()
    env.add_state_values(result.values, precision=args.precision)
    env.add_policy(result.policy)
    env.render(animation_interval=2)


if __name__ == "__main__":
    main()
