# Bluffbot (Poker RL)

A heads-up (1v1) No-Limit Hold'em RL agent trained via self-play with PPO. Built after playing a few poker games this summer.

## Layout

```
env/
  deck.py            card dealing + one-hot encoding (wraps treys)
  hand_eval.py        hand_strength(hole_cards, board) -> float in [0, 1]
  poker_env.py         HeadsUpPokerEnv (gymnasium.Env): core game rules
  self_play_env.py    SelfPlayEnv: wraps HeadsUpPokerEnv so PPO always controls
                       player 0; the opponent's turns are played automatically
agents/
  random_agent.py      uniform-random baseline
  always_call_agent.py fixed baseline, always calls
  tight_agent.py        fixed baseline, folds/calls/raises off hand strength
  ppo_agent.py          wraps stable-baselines3 PPO (predict/train/save/load)
  opponent_pool.py     holds past PPO snapshots, samples one as the self-play
                        opponent each episode (fictitious-self-play style)
training/
  eval.py               evaluate(agent, opponent, n_hands) -> win_rate, avg_reward;
                        bb_per_100() converts avg_reward to bb/100 hands
  action_distribution.py tallies which actions an agent takes over N hands
  self_play.py         main training loop
  hyperparam_sweep.py  compares a few PPO hyperparameter configs
```

## Setup

```
python -m venv venv
source venv/bin/activate
pip install stable_baselines3 gymnasium torch treys numpy matplotlib
```

## Usage

Train (self-play PPO vs an opponent pool, evaluated each round against
random / always-call / tight baselines):

```
python -m training.self_play
```

Saves `trained_agent_selfplay.zip` and `training_curve_selfplay.png` (win rate,
avg reward, and bb/100 per baseline).

Check what actions the trained agent actually takes (useful for catching a
policy that's collapsed onto always raising / always folding):

```
python -m training.action_distribution
```

Compare a few PPO hyperparameter configs (learning rate, entropy coefficient):

```
python -m training.hyperparam_sweep
```

## Notes

- The action space is 6 fixed actions: fold, call, raise 0.5x/1x/2x pot, or
  all-in — not a continuous bet size.
- Effective stacks are shallow (50 big blinds), so bb/100 win-rate figures run
  much larger than real-poker benchmarks. Useful for comparing configs against
  each other, not against real-poker win rates.
- `TightAgent` is a single-threshold heuristic (folds below 0.4 hand strength,
  raises above 0.75, always calls preflop) — a decent "opponent that punishes
  shoving," not a strong poker bot. It's the best of the three fixed baselines
  for catching a degenerate all-in/raise-heavy policy, since `RandomAgent` and
  `AlwaysCallAgent` can't fold in response to bad play.
