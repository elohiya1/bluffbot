from training.self_play import train
from training.action_distribution import action_distribution
from agents.tight_agent import TightAgent

RAISE_ACTIONS = {2, 3, 4, 5}  # RAISE_HALF, RAISE_FULL, RAISE_2X, ALL_IN

CONFIGS = {
    "default": {},                        # lr=1e-4, ent_coef=0.01 (train()'s current defaults)
    "higher_lr": {"learning_rate": 3e-4},  # original lr, for regression comparison
    "no_entropy": {"ent_coef": 0.0},       # isolates the entropy bonus's contribution
}


def run_sweep(total_rounds=5, hands_per_round=5_000):
    results = {}

    for name, overrides in CONFIGS.items():
        print(f"\n=== Config: {name} ({overrides}) ===")
        result = train(total_rounds=total_rounds, hands_per_round=hands_per_round, tag=name, **overrides)
        agent = result["agent"]

        counts_vs_random = action_distribution(agent, n_hands=500)
        counts_vs_tight = action_distribution(agent, n_hands=500, opponent=TightAgent())
        total_random = sum(counts_vs_random.values())
        total_tight = sum(counts_vs_tight.values())
        raise_pct_random = sum(counts_vs_random[a] for a in RAISE_ACTIONS) / total_random * 100
        raise_pct_tight = sum(counts_vs_tight[a] for a in RAISE_ACTIONS) / total_tight * 100

        results[name] = {
            "final_wr_random": result["win_rates"]["random"][-1],
            "final_wr_call": result["win_rates"]["always_call"][-1],
            "final_wr_tight": result["win_rates"]["tight"][-1],
            "final_bb100_random": result["bb100s"]["random"][-1],
            "final_bb100_call": result["bb100s"]["always_call"][-1],
            "final_bb100_tight": result["bb100s"]["tight"][-1],
            "raise_pct_random": raise_pct_random,
            "raise_pct_tight": raise_pct_tight,
        }

    print("\n=== Sweep comparison ===")
    header = (
        f"{'config':12s} {'wr random':10s} {'wr call':9s} {'wr tight':9s} "
        f"{'bb/100 rand':12s} {'bb/100 call':12s} {'bb/100 tight':12s} "
        f"{'raise% (rand)':14s} {'raise% (tight)':14s}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:12s} {r['final_wr_random']:<10.2f} {r['final_wr_call']:<9.2f} {r['final_wr_tight']:<9.2f} "
            f"{r['final_bb100_random']:<12.1f} {r['final_bb100_call']:<12.1f} {r['final_bb100_tight']:<12.1f} "
            f"{r['raise_pct_random']:<14.1f} {r['raise_pct_tight']:<14.1f}"
        )

    return results


if __name__ == "__main__":
    run_sweep()
