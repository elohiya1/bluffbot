from env.poker_env import FOLD, CALL, RAISE_FULL

# Indices into the obs vector (52 hole one-hot + 260 board one-hot + 8 scalars).
_SCALARS_START = 52 + 260
STRENGTH_IDX = _SCALARS_START + 3
BOARD_LEN_IDX = _SCALARS_START + 6

FOLD_THRESHOLD = 0.4
RAISE_THRESHOLD = 0.75


class TightAgent:
    """
    Fixed, non-learning baseline that actually folds weak hands.
    Calls preflop (no hand-strength signal yet), then postflop folds weak
    hands, calls medium hands, and value-raises strong hands. Unlike
    RandomAgent/AlwaysCallAgent, this can punish an opponent that shoves
    regardless of hand strength.
    """

    def act(self, obs):
        preflop = obs[BOARD_LEN_IDX] == 0.0
        if preflop:
            return CALL

        strength = obs[STRENGTH_IDX]
        if strength < FOLD_THRESHOLD:
            return FOLD
        if strength >= RAISE_THRESHOLD:
            return RAISE_FULL
        return CALL
