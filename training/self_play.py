import random
import tempfile
import os
import matplotlib.pyplot as plt
from env.self_play_env import SelfPlayEnv
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from agents.always_call_agent import AlwaysCallAgent
from agents.tight_agent import TightAgent
from agents.opponent_pool import OpponentPool
from training.eval import evaluate, bb_per_100
from stable_baselines3 import PPO as SB3PPO

BASELINES = {
    "random": RandomAgent,
    "always_call": AlwaysCallAgent,
    "tight": TightAgent,
}

def train(
    total_rounds=10,
    hands_per_round=5_000,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    pool_max_size=10,
    tag="selfplay",
):
    steps_per_round = hands_per_round * 10

    # Start with random opponent
    opponent_policy = lambda obs: random.randint(0, 5)
    env = SelfPlayEnv(opponent_policy=opponent_policy)
    pool = OpponentPool(max_size=pool_max_size)

    model = SB3PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        verbose=0
    )

    win_rates = {name: [] for name in BASELINES}
    avg_rewards = {name: [] for name in BASELINES}
    bb100s = {name: [] for name in BASELINES}
    checkpoints = []

    def run_eval(agent, round_num):
        parts = []
        for name, agent_cls in BASELINES.items():
            wr, ar = evaluate(agent, opponent=agent_cls(), n_hands=500)
            bb100 = bb_per_100(ar)
            win_rates[name].append(wr)
            avg_rewards[name].append(ar)
            bb100s[name].append(bb100)
            parts.append(f"vs {name} win rate={wr:.2f}, avg reward={ar:.4f}, bb/100={bb100:.1f}")
        checkpoints.append(round_num * hands_per_round)
        print(f"  Round {round_num}: " + " | ".join(parts))

    # Evaluate untrained
    agent = PPOAgent(model=model)
    print("Evaluating untrained agent...")
    run_eval(agent, 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        for round_num in range(1, total_rounds + 1):
            print(f"\nRound {round_num}/{total_rounds}: training for {hands_per_round} hands...")

            # Train
            model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)

            # Snapshot current policy into the opponent pool (deepcopy doesn't work on
            # SB3's internal torch state, so snapshot via save/load instead)
            snapshot_path = os.path.join(tmpdir, f"snapshot_round_{round_num}")
            model.save(snapshot_path)
            pool.add_snapshot(snapshot_path)
            env.set_opponent_pool(pool)

            # Evaluate against all baselines
            agent = PPOAgent(model=model)
            run_eval(agent, round_num)

    # Save
    agent.save(f"trained_agent_{tag}")
    print(f"\nAgent saved to trained_agent_{tag}.zip")

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    for name in BASELINES:
        ax1.plot(checkpoints, win_rates[name], marker='o', label=f'vs {name}')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='50% baseline')
    ax1.set_xlabel("Hands trained")
    ax1.set_ylabel("Win rate")
    ax1.set_title("Win Rate Over Self-Play Training")
    ax1.legend()

    for name in BASELINES:
        ax2.plot(checkpoints, avg_rewards[name], marker='o', label=f'vs {name}')
    ax2.axhline(y=0.0, color='r', linestyle='--', label='break even')
    ax2.set_xlabel("Hands trained")
    ax2.set_ylabel("Avg reward per hand")
    ax2.set_title("Average Reward Over Self-Play Training")
    ax2.legend()

    for name in BASELINES:
        ax3.plot(checkpoints, bb100s[name], marker='o', label=f'vs {name}')
    ax3.axhline(y=0.0, color='r', linestyle='--', label='break even')
    ax3.set_xlabel("Hands trained")
    ax3.set_ylabel("bb/100 hands")
    ax3.set_title("Win Rate in bb/100 (standard poker unit)")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"training_curve_{tag}.png")
    print(f"Training curve saved to training_curve_{tag}.png")

    return {
        "agent": agent,
        "win_rates": win_rates,
        "avg_rewards": avg_rewards,
        "bb100s": bb100s,
        "checkpoints": checkpoints,
    }

if __name__ == "__main__":
    train()
