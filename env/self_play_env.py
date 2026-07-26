import numpy as np
import gymnasium as gym
from env.poker_env import HeadsUpPokerEnv, NUM_ACTIONS
import random

class SelfPlayEnv(gym.Env):
    """
    Wraps HeadsUpPokerEnv so PPO only controls player 0.
    Player 1's actions are taken automatically by an opponent policy.
    The opponent can be swapped out between training rounds.
    """

    def __init__(self, opponent_policy=None):
        super().__init__()
        self.env = HeadsUpPokerEnv()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # opponent_policy is a callable: obs -> action
        # if None, opponent plays randomly
        self.opponent_policy = opponent_policy or (lambda obs: random.randint(0, NUM_ACTIONS - 1))
        self._opponent_provider = lambda: self.opponent_policy

    def set_opponent(self, policy):
        """Swap in a single fixed opponent policy used for every episode."""
        self.opponent_policy = policy
        self._opponent_provider = lambda: self.opponent_policy

    def set_opponent_pool(self, pool):
        """Sample a fresh opponent policy from the pool at the start of each episode."""
        self._opponent_provider = pool.sample_policy

    def reset(self, seed=None, options=None):
        self.opponent_policy = self._opponent_provider()
        obs, info = self.env.reset()
        # If it's opponent's turn first, let them act
        obs = self._play_opponent_turns(obs)
        return obs, info

    def step(self, action):
        # Player 0 acts
        obs, reward, done, truncated, info = self.env.step(action)

        # Now let opponent act until it's player 0's turn again or hand ends
        if not done:
            obs = self._play_opponent_turns(obs)

        # Re-check done after opponent acts
        done = self.env.done
        reward = self._get_reward()

        return obs, reward, done, truncated, info

    def _play_opponent_turns(self, obs):
        """Keep acting for player 1 until it's player 0's turn or hand ends."""
        while not self.env.done and self.env.current_player == 1:
            action = self.opponent_policy(obs)
            obs, _, _, _, _ = self.env.step(action)
        return obs

    def _get_reward(self):
        """Return reward from player 0's perspective."""
        from env.poker_env import STARTING_STACK
        if not self.env.done:
            return 0.0
        return (self.env.stacks[0] - STARTING_STACK) / STARTING_STACK