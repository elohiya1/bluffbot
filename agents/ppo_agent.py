from stable_baselines3 import PPO
from env.poker_env import HeadsUpPokerEnv

class PPOAgent:
    """
    Wraps stable-baselines3 PPO.
    Handles training and action selection.
    """

    def __init__(self, model=None):
        if model is not None:
            self.model = model
        else:
            env = HeadsUpPokerEnv()
            self.model = PPO(
                policy="MlpPolicy",      # multi-layer perceptron -- a standard feedforward neural net
                env=env,
                learning_rate=3e-4,      # how fast the net updates its weights
                n_steps=2048,            # how many steps to collect before each update
                batch_size=64,           # how many samples to use per gradient update
                n_epochs=10,             # how many times to reuse each batch
                gamma=0.99,              # discount factor for future rewards
                verbose=0
            )

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path):
        model = PPO.load(path)
        return cls(model=model)