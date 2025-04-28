
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import random
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from hand import RANKS, SUITS, embed_with_community
from phevaluator.evaluator import evaluate_cards

suits = ['d','s','c','h']
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
cards = []
for r in ranks:
    for s in suits:
        cards.append(r+s)
        
def hand_strength(hole_cards: List[str], community_cards: List[str], players: int = 2) -> int:
    hands = []
    deck = random.sample(cards,len(cards))
    hole_cards = [card[1] + card[0].lower() for card in hole_cards]
    community_cards = [card[1] + card[0].lower() for card in community_cards]
    full = community_cards + hole_cards
    deck = list(filter(lambda x: x not in full, deck))
    for i in range(players):
        hn = []
        hn.append(deck[0])
        deck = deck[1:]
        hn.append(deck[0])
        deck = deck[1:]
        hands.append(hn)
    while len(community_cards) < 5:
        card = deck.pop(0)
        community_cards.append(card)
        full.append(card)
    my_hand_rank = evaluate_cards(full[0],full[1],full[2],full[3],full[4],full[5],full[6])

    for check_hand in hands:
        all_cards = community_cards + check_hand
        opponent = evaluate_cards(all_cards[0],all_cards[1],all_cards[2],all_cards[3],all_cards[4],all_cards[5],all_cards[6])
        if opponent < my_hand_rank:
            return 1
        if opponent == my_hand_rank:
            return 2
        return 0

def estimate_hand_strength(hole_cards: List[str], community_cards: List[str], players: int = 2, simulations: int = 100) -> np.ndarray:
    # wins, losses, ties
    outcomes = np.zeros(3, dtype=np.float32)
    for _ in range(simulations):
        outcome = hand_strength(hole_cards, community_cards, players)
        outcomes[outcome] += 1
    return outcomes / simulations

def sample_hand_data(_, simulations: int) -> Tuple[List[float], np.float32]:
    deck = [s + r for r in RANKS for s in SUITS]
    np.random.shuffle(deck)
    hole_cards = deck[:2]
    street = np.random.randint(4)

    if street == 0: # preflop
        community_cards = []
    elif street == 1: # flop
        community_cards = deck[2:5]
    elif street == 2: # turn
        community_cards = deck[2:6]
    else:  # river
        community_cards = deck[2:7]

    features = embed_with_community(hole_cards, community_cards)
    wins, _, _ = estimate_hand_strength(hole_cards, community_cards, simulations=simulations)
    return street, features, wins

def generate_hand_equity(n_hands: int = 1_000_000, simulations: int = 10_000):
    num_workers = cpu_count()
    print(f"Generating training data using {num_workers} workers...")

    wrapped_func = partial(sample_hand_data, simulations=simulations)
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(wrapped_func, range(n_hands)), total=n_hands))

    X_cat = np.array([res[0] for res in results], dtype=np.uint8)
    X = np.array([res[1] for res in results], dtype=np.float32)
    y = np.array([res[2] for res in results], dtype=np.float32)

    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "hand_features.npy", X)
    np.save(data_dir / "hand_categorical.npy", X_cat)
    np.save(data_dir / "hand_targets.npy", y)

    return X_cat, X, y