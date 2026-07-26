from treys import Evaluator

evaluator = Evaluator()

WORST_SCORE = 7462  # treys score for the worst possible hand
BEST_SCORE = 1      # treys score for a royal flush

def hand_strength(hole_cards, board):
    """
    Returns a float between 0 and 1.
    1.0 = best possible hand (royal flush)
    0.0 = worst possible hand
    Returns 0.0 if board has fewer than 3 cards (preflop/no eval possible).
    """
    if len(board) < 3:
        return 0.0

    score = evaluator.evaluate(board, hole_cards)
    
    # treys is backwards: lower score = better hand
    # we flip it so that higher number = stronger hand
    strength = 1.0 - (score - BEST_SCORE) / (WORST_SCORE - BEST_SCORE)
    return strength