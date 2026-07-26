from env.poker_env import HeadsUpPokerEnv, STARTING_STACK, BIG_BLIND
from agents.random_agent import RandomAgent

def bb_per_100(avg_reward):
    """
    Converts avg_reward (fraction of starting stack won per hand) into
    bb/100 hands, the standard poker win-rate unit, so results are
    comparable to normal poker benchmarks (e.g. 5-10 bb/100 is a solid
    winrate; a toy-env exploit often shows up as an unrealistically high
    number here).
    """
    return avg_reward * (STARTING_STACK / BIG_BLIND) * 100

def evaluate(agent, opponent=None, n_hands=500, verbose=False):
    """
    Runs agent (player 0) against opponent (player 1, defaults to random) for n_hands.
    Returns win rate and average reward per hand.
    """
    env = HeadsUpPokerEnv()
    opponent = opponent or RandomAgent()

    wins = 0
    total_reward = 0.0

    for hand in range(n_hands):
        obs, _ = env.reset()
        done = False

        while not done:
            if env.current_player == 0:
                action = agent.act(obs)
            else:
                action = opponent.act(obs)

            obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        if reward > 0:
            wins += 1

        if verbose and hand % 100 == 0:
            print(f"Hand {hand}: reward={reward:.3f}, running win rate={wins/(hand+1):.2f}")

    win_rate = wins / n_hands
    avg_reward = total_reward / n_hands
    return win_rate, avg_reward