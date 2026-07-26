from collections import Counter
from env.poker_env import HeadsUpPokerEnv
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent

ACTION_NAMES = ["FOLD", "CALL", "RAISE_HALF", "RAISE_FULL", "RAISE_2X", "ALL_IN"]


def action_distribution(agent, n_hands=1000, opponent=None):
    env = HeadsUpPokerEnv()
    opponent = opponent or RandomAgent()
    counts = Counter()

    for _ in range(n_hands):
        obs, _ = env.reset()
        done = False
        while not done:
            if env.current_player == 0:
                action = agent.act(obs)
                counts[action] += 1
            else:
                action = opponent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)

    total = sum(counts.values())
    print(f"Action distribution over {total} agent decisions ({n_hands} hands):")
    for i, name in enumerate(ACTION_NAMES):
        c = counts[i]
        print(f"  {name:12s}: {c:6d} ({c/total*100:5.1f}%)")
    return counts


if __name__ == "__main__":
    agent = PPOAgent.load("trained_agent_selfplay")
    action_distribution(agent, n_hands=1000)
