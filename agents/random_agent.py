import random

class RandomAgent:
    """Picks a random action every time. Used as a baseline opponent."""

    def __init__(self, num_actions=6):
        self.num_actions = num_actions

    def act(self, obs):
        return random.randint(0, self.num_actions - 1)