import torch
import torch.nn as nn

def card_to_num(card_str):
    """
    Convert a card string (e.g., 'S8', 'HQ') to a tuple of (rank, suit).

    Args:
        card_str: String representation of a card
                 T=10, J=Jack, Q=Queen, K=King, A=Ace
                 C=Clubs, D=Diamonds, H=Hearts, S=Spades

    Returns:
        tuple: (rank, suit) where:
               rank is 0-12 (0=2, 1=3, ..., 12=Ace)
               suit is 0-3 (0=clubs, 1=diamonds, 2=hearts, 3=spades)
    """
    suit_str = "CDHS"
    rank_str = "23456789TJQKA"
    suit = suit_str.index(card_str[0])
    rank = rank_str.index(card_str[1])
    return (rank, suit)


# Straight potential calculation
def calc_straight_potential(ranks):
    if not ranks:
        return 0.0
    # Count consecutive ranks
    unique_ranks = sorted(set(ranks))
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(unique_ranks)):
        if unique_ranks[i] == unique_ranks[i-1] + 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    # Also check for wheel straight (A-2-3-4-5)
    if 14 in unique_ranks and {2, 3, 4, 5} & set(unique_ranks):
        wheel_count = 1 + sum(1 for r in [2, 3, 4, 5] if r in unique_ranks)
        max_consecutive = max(max_consecutive, wheel_count)
    return min(max_consecutive / 5.0, 1.0)

# Flush potential calculation
def calc_flush_potential(suits):
    if not suits:
        return 0.0
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    return min(max(suit_counts.values()) / 5.0, 1.0)

def street_to_num(street):
    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    return street_map.get(street, -1)

def encode(hole_cards, community_cards, street: str, pot_size, stack, opponent_stack, round_count, is_small_blind):
    """
    Encode a poker hand (hole cards + available community cards) into a fixed-length feature vector.

    Parameters:
    - hole_cards: List of 2 cards, each represented as (rank, suit)
    - community_cards: List of 0-5 community cards, each represented as (rank, suit)
    - street: preflop, flop, turn, river
    - pot_size: Int
    - stack
    - opponent_stack
    - round_count
    - is_small_blind (0 or 1)

    Returns a list of features each normalized item representing:
    - high card (hole_cards)
    - low card (hole_cards)
    - high suit rank (hole_cards)
    - low suit rank (hole_cards)
    - is suited (hole_cards)
    - is pair (hole_cards)
    - number of cards available (community_cards)
    - pairs (0-2) (community_cards)
    - trips (0-1) (community_cards)
    - straight potential (community_cards)
    - flush potential (community_cards)
    - pairs (0-2) (all_cards)
    - trips (0-1) (all_cards)
    - straight potential (all_cards)
    - flush potential (all_cards)
    - street (0-3)
    - stack
    - opponent_stack
    - round_count
    - is_small_blind (0 or 1)

    """
    hole_cards = list(map(card_to_num, hole_cards))
    community_cards = list(map(card_to_num, community_cards))

    # Extract hole cards
    card1, card2 = hole_cards
    rank1, suit1 = card1
    rank2, suit2 = card2

    # Basic hole card features
    high_card = max(rank1, rank2) / 12  # Normalized high card
    low_card = min(rank1, rank2) / 12   # Normalized low card
    high_suit = max(suit1, suit2) / 4  # Normalized high suit
    low_suit = min(suit1, suit2) / 4   # Normalized low suit
    suited = 1.0 if suit1 == suit2 else 0.0  # Whether cards are suited
    pair = 1.0 if rank1 == rank2 else 0.0    # Whether cards are a pair

    # Community cards features
    num_community = len(community_cards) / 5.0  # Normalize number of community cards

    # Process community cards
    community_ranks = [r for r, _ in community_cards]
    community_suits = [s for _, s in community_cards]

    # Count rank occurrences in community cards
    rank_counts_community = {}
    for rank in community_ranks:
        rank_counts_community[rank] = rank_counts_community.get(rank, 0) + 1

    # Pairs and trips in community cards
    pairs_community = min(sum(1 for count in rank_counts_community.values() if count == 2), 2) / 2.0
    trips_community = min(sum(1 for count in rank_counts_community.values() if count >= 3), 1) / 1.0

    # Process all cards
    all_ranks = [rank1, rank2] + community_ranks
    all_suits = [suit1, suit2] + community_suits

    # Count rank occurrences in all cards
    rank_counts_all = {}
    for rank in all_ranks:
        rank_counts_all[rank] = rank_counts_all.get(rank, 0) + 1

    # Pairs and trips in all cards
    pairs_all = min(sum(1 for count in rank_counts_all.values() if count == 2), 2) / 2.0
    trips_all = min(sum(1 for count in rank_counts_all.values() if count >= 3), 1) / 1.0


    # Calculate straight and flush potentials
    straight_potential_community = calc_straight_potential(community_ranks)
    flush_potential_community = calc_flush_potential(community_suits)
    straight_potential_all = calc_straight_potential(all_ranks)
    flush_potential_all = calc_flush_potential(all_suits)

    # Combine all features
    features = [
        # hole cards
        high_card,
        low_card,
        suited,
        pair,
        # community
        num_community,
        pairs_community,
        trips_community,
        straight_potential_community,
        flush_potential_community,
        # all
        pairs_all,
        trips_all,
        straight_potential_all,
        flush_potential_all,
        # others,
        street_to_num(street) / 3,
        stack / 2000,
        opponent_stack / 2000,
        round_count / 100,
        is_small_blind,
        pot_size / 2000
    ]

    return features