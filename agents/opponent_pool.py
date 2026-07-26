import random
from stable_baselines3 import PPO

class OpponentPool:
    """Holds past PPO snapshots and samples one as an opponent policy per episode."""

    def __init__(self, max_size=10):
        self.max_size = max_size
        self.models = []

    def add_snapshot(self, path):
        model = PPO.load(path)
        self.models.append(model)
        if len(self.models) > self.max_size:
            self.models.pop(0)

    def sample_policy(self):
        model = random.choice(self.models)
        def policy(obs):
            action, _ = model.predict(obs, deterministic=False)
            return int(action)
        return policy
