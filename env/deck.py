from treys import Card, Deck

def new_deck():
    """Returns a shuffled list of 52 card ints."""
    deck = Deck()
    deck.shuffle()
    return deck.cards[:]

def card_to_index(card_int):
    """Maps a treys card int to a 0-51 index."""
    rank = Card.get_rank_int(card_int)        # 0-12  (2=0, 3=1, ... A=12)
    suit = Card.get_suit_int(card_int)        # 1, 2, 4, or 8
    suit_idx = {1: 0, 2: 1, 4: 2, 8: 3}[suit]
    return rank * 4 + suit_idx

def cards_to_onehot(card_list):
    """Converts a list of treys card ints to a 52-dim one-hot vector."""
    vec = [0.0] * 52
    for card in card_list:
        vec[card_to_index(card)] = 1.0
    return vec