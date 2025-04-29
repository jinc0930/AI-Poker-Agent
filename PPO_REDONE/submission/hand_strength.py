from functools import lru_cache
import random
from typing import Final, List, Tuple
from collections import Counter
from typing import List, Tuple

SUITS: Final[str] = "CDHS"
RANKS: Final[str] = "23456789TJQKA"
STREET_MAP: Final[dict] = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}

def generate_cards() -> Tuple[List[str], List[str]]:
    deck = [s + r for s in SUITS for r in RANKS]
    random.shuffle(deck)
    hole_cards = deck[:2]
    num_community_cards = random.choice([0, 3, 4, 5])
    community_cards = deck[2:2 + num_community_cards]
    return hole_cards, community_cards

def card_to_num(card_str: str) -> Tuple[int, int]:
    """Returns (rank, suit)"""
    # Handle two-character and three-character card strings
    if len(card_str) == 2:
        suit = SUITS.index(card_str[0])
        rank = RANKS.index(card_str[1])
    elif len(card_str) == 3:
        suit = SUITS.index(card_str[0])
        rank = int(card_str[1:])
    else:
        raise ValueError(f"Invalid card string: {card_str}")

    return (rank, suit)

def street_to_num(street: str) -> int:
    return STREET_MAP[street]

def flush_potential(hole_cards: List[str], community_cards: List[str]) -> Tuple[int, int]:
    """Measure flush potential and count of hole cards contributing to it"""
    hole_suits = [card_to_num(card)[1] for card in hole_cards]
    comm_suits = [card_to_num(card)[1] for card in community_cards]
    all_suits = hole_suits + comm_suits
    suit_counts = {}
    for suit in all_suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    if not suit_counts:
        return 0, 0

    max_suit = max(suit_counts, key=suit_counts.get)
    flush_potential = suit_counts[max_suit]
    contributing_hole_cards = hole_suits.count(max_suit)
    return flush_potential, contributing_hole_cards

def straight_potential(hole_cards: List[str], community_cards: List[str]) -> Tuple[int, int]:
    """Measure card connectedness for straight potential."""
    hole_ranks = [card_to_num(card)[0] for card in hole_cards]
    comm_ranks = [card_to_num(card)[0] for card in community_cards]
    all_ranks = sorted(set(hole_ranks + comm_ranks))
    if 12 in all_ranks: # add before ace
        all_ranks = [-1] + all_ranks

    best_gap_count = float('inf')
    best_consecutive = 0
    best_window = []

    for start in range(len(all_ranks)):
        end = start
        while end < len(all_ranks) and all_ranks[end] - all_ranks[start] < 5:
            end += 1

        window = all_ranks[start:end]

        # skip len 2
        if len(window) < 2:
            continue

        # count gaps and consecutive cards
        total_gaps = window[-1] - window[0] + 1 - len(window)
        longest_seq = 1
        current_seq = 1

        for i in range(1, len(window)):
            if window[i] == window[i-1] + 1:
                current_seq += 1
                longest_seq = max(longest_seq, current_seq)
            else:
                current_seq = 1

        if (longest_seq > best_consecutive or (longest_seq == best_consecutive and total_gaps < best_gap_count)):
            best_consecutive = longest_seq
            best_gap_count = total_gaps
            best_window = window

    if not best_window:
        return (0, 0)

    if best_consecutive >= 5:
        score = 5 # straight
    elif best_consecutive == 4:
        score = 4 # open-ended draw
    elif best_consecutive == 3:
        if best_gap_count == 1:
            score = 3 # outshot
        elif best_gap_count == 2:
            score = 2 # double gutshot
        else:
            score = 1
    elif best_consecutive == 2:
        score = 1
    else:
        score = 0
    hole_cards_used = sum(1 for r in best_window if r in hole_ranks or (r == -1 and 12 in hole_ranks))

    return (score, hole_cards_used)

def hole_card_connectivity(card1: str, card2: str) -> float:
    r1 = card_to_num(card1)[0]
    r2 = card_to_num(card2)[0]
    diff = abs(r1 - r2)
    alt_diff = 0
    if r1 == 12 or r2 == 12: # if ace is involved
        alt_diff = abs((0 if r1 == 12 else r1) - (0 if r2 == 12 else r2))
    return min(diff, alt_diff)

def count_ranks(hole_cards: List[str], community_cards: List[str]) -> dict[str, list[int]]:
    """Count pairs, trips, quads, full houses and hole card contributions"""
    hole_ranks = [card_to_num(card)[0] for card in hole_cards]
    comm_ranks = [card_to_num(card)[0] for card in community_cards]

    hole_counter = Counter(hole_ranks)
    comm_counter = Counter(comm_ranks)
    all_counter = hole_counter + comm_counter

    results = {
        'pairs': [0, 0],
        'trips': [0, 0],
        'quads': [0, 0],
        'full_house': [0, 0]
    }
    trips = []
    pairs = []
    for rank, count in all_counter.items():
        hole_count = hole_counter[rank]

        if count == 2:
            results['pairs'][0] += 1
            results['pairs'][1] += min(hole_count, 2)
            pairs.append(rank)
        elif count == 3:
            results['trips'][0] += 1
            results['trips'][1] += min(hole_count, 3)
            trips.append(rank)
        elif count == 4:
            results['quads'][0] += 1
            results['quads'][1] += min(hole_count, 4)

    if (trips and pairs) or len(trips) >= 2:
        results['full_house'][0] = 1

        if len(trips) >= 2:
            trips.sort(reverse=True)
            results['full_house'][1] = min(hole_counter[trips[0]], 3) + min(hole_counter[trips[1]], 2)
        else:
            valid_pairs = [p for p in pairs if p != trips[0]]
            if valid_pairs:
                pair_rank = max(valid_pairs)
                results['full_house'][1] = min(hole_counter[trips[0]], 3) + min(hole_counter[pair_rank], 2)

    return results

@lru_cache(maxsize=10_000)
def embed_hole_cards(card1: str, card2: str) -> List[float]:
    """Base hole card embeddings, dim = 5"""

    rank1, suit1 = card_to_num(card1)
    rank2, suit2 = card_to_num(card2)

    high_rank = max(rank1, rank2)
    low_rank = min(rank1, rank2)

    is_suited = 1 if suit1 == suit2 else 0
    is_pair = 1 if rank1 == rank2 else 0
    connectedness = hole_card_connectivity(card1, card2)

    max_rank = len(RANKS) - 1
    return [high_rank/max_rank, low_rank/max_rank, is_suited, is_pair, connectedness]

def embed_with_community(hole_cards: List[str], community_cards: List[str]) -> List[float]:
    """Embed both hole cards with community, dim = 17"""
    card1, card2 = hole_cards
    base_embedding = embed_hole_cards(card1, card2)
    ranks = count_ranks(hole_cards, community_cards)
    flush, flush_p = flush_potential(hole_cards, community_cards)
    straight, straight_p = straight_potential(hole_cards, community_cards)
    community_features = [flush/5, flush_p/2, straight/5, straight_p/2]
    ranks = [
        ranks['pairs'][0]/3,
        ranks['pairs'][1]/2,
        ranks['trips'][0]/2,
        ranks['trips'][1]/2,
        ranks['quads'][0],
        ranks['quads'][1]/2,
        ranks['full_house'][0],
        ranks['full_house'][1]/2,
    ]
    return base_embedding + community_features + ranks
