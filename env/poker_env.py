import numpy as np
import gymnasium as gym
from gymnasium import spaces
from treys import Card, Deck
from env.deck import new_deck, cards_to_onehot
from env.hand_eval import hand_strength

# --- Constants ---
STARTING_STACK = 100
SMALL_BLIND = 1
BIG_BLIND = 2

# Actions
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_FULL = 3
RAISE_2X = 4
ALL_IN = 5

NUM_ACTIONS = 6


class HeadsUpPokerEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(NUM_ACTIONS)

        obs_size = 52 + 260 + 8
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.deck = []
        self.hole_cards = [[], []]
        self.board = []
        self.stacks = [STARTING_STACK, STARTING_STACK]
        self.pot = 0
        self.current_player = 0
        self.street = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.deck = new_deck()

        self.hole_cards[0] = [self.deck.pop(), self.deck.pop()]
        self.hole_cards[1] = [self.deck.pop(), self.deck.pop()]

        self.board = []
        self.stacks = [STARTING_STACK, STARTING_STACK]
        self.pot = 0
        self.street = 0
        self.done = False
        self.current_player = 0

        self._post_blinds()

        return self._get_obs(), {}

    def _post_blinds(self):
        sb = min(SMALL_BLIND, self.stacks[0])
        self.stacks[0] -= sb
        self.pot += sb

        bb = min(BIG_BLIND, self.stacks[1])
        self.stacks[1] -= bb
        self.pot += bb

    def _get_obs(self):
        # Cards/strength are relative to current_player so each player sees
        # its own hand, not always seat 0's (needed for self-play to be meaningful).
        hole_onehot = cards_to_onehot(self.hole_cards[self.current_player])

        board_onehot = []
        for i in range(5):
            if i < len(self.board):
                board_onehot.extend(cards_to_onehot([self.board[i]]))
            else:
                board_onehot.extend([0.0] * 52)

        strength = hand_strength(self.hole_cards[self.current_player], self.board)
        scalars = [
            self.pot / (STARTING_STACK * 2),
            self.stacks[0] / STARTING_STACK,
            self.stacks[1] / STARTING_STACK,
            strength,
            self.street / 3.0,
            float(self.current_player),
            float(len(self.board)) / 5.0,
            float(self.done),
        ]

        obs = hole_onehot + board_onehot + scalars
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        assert not self.done, "Hand is over. Call reset() to start a new hand."

        reward = 0.0
        truncated = False

        if action == FOLD:
            winner = 1 - self.current_player
            reward = self._resolve(winner)
            self.done = True

        elif action == CALL:
            amount = min(BIG_BLIND, self.stacks[self.current_player])
            self.stacks[self.current_player] -= amount
            self.pot += amount
            self._advance_street()

        elif action in (RAISE_HALF, RAISE_FULL, RAISE_2X, ALL_IN):
            raise_amount = self._get_raise_amount(action)
            if raise_amount == 0:
                # No chips left to raise with (already all-in) -- treat as a
                # call so the street advances instead of volleying zero-chip
                # "raises" back and forth indefinitely.
                amount = min(BIG_BLIND, self.stacks[self.current_player])
                self.stacks[self.current_player] -= amount
                self.pot += amount
                self._advance_street()
            else:
                self.stacks[self.current_player] -= raise_amount
                self.pot += raise_amount
                self.current_player = 1 - self.current_player

        if not self.done and self.street > 3:
            winner = self._showdown()
            reward = self._resolve(winner)
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, truncated, {}

    def _get_raise_amount(self, action):
        if action == ALL_IN:
            return self.stacks[self.current_player]

        multipliers = {
            RAISE_HALF: 0.5,
            RAISE_FULL: 1.0,
            RAISE_2X: 2.0,
        }
        amount = int(self.pot * multipliers[action])
        return min(amount, self.stacks[self.current_player])

    def _advance_street(self):
        self.street += 1
        self.current_player = 0

        if self.street == 1:
            self.board.append(self.deck.pop())
            self.board.append(self.deck.pop())
            self.board.append(self.deck.pop())
        elif self.street in (2, 3):
            self.board.append(self.deck.pop())

    def _showdown(self):
        from treys import Evaluator
        evaluator = Evaluator()
        score0 = evaluator.evaluate(self.board, self.hole_cards[0])
        score1 = evaluator.evaluate(self.board, self.hole_cards[1])
        return 0 if score0 <= score1 else 1

    def _resolve(self, winner):
        self.stacks[winner] += self.pot
        reward = (self.stacks[0] - STARTING_STACK) / STARTING_STACK
        self.pot = 0
        return reward