from env.poker_env import CALL

class AlwaysCallAgent:
    """Always calls. Never folds or raises. A fixed, non-learning baseline."""

    def act(self, obs):
        return CALL
