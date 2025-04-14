import random
import numpy as np
from typing import List
from phevaluator import evaluate_cards

suits = ['d','s','c','h']
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
cards = []
for r in ranks:
    for s in suits:
        cards.append(r+s)

def simulate(hand: List[str], table:List[str], players = 2):
    hands = []
    deck = random.sample(cards,len(cards))
    hand = [card[1] + card[0].lower() for card in hand]
    table = [card[1] + card[0].lower() for card in table]
    full = table + hand
    deck = list(filter(lambda x: x not in full, deck))
    for i in range(players):
        hn = []
        hn.append(deck[0])
        deck = deck[1:]
        hn.append(deck[0])
        deck = deck[1:]
        hands.append(hn)
    while len(table) < 5:
        card = deck.pop(0)
        table.append(card)
        full.append(card)
    my_hand_rank = evaluate_cards(full[0],full[1],full[2],full[3],full[4],full[5],full[6])
    for check_hand in hands:
        all_cards = table + check_hand
        opponent = evaluate_cards(all_cards[0],all_cards[1],all_cards[2],all_cards[3],all_cards[4],all_cards[5],all_cards[6])
        if opponent < my_hand_rank:
            return 1
        if opponent == my_hand_rank:
            return 2
        return 0

def monte_carlo(hand, table, players=2, samples: int =100):
    outcomes = np.zeros(3, dtype=np.int32)
    for _ in range(samples):
        outcome = simulate(hand, table, players)
        outcomes[outcome] += 1
    return outcomes / samples

def get_stacks(seats, your_uuid):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

def is_big_blind(action_histories, player_uuid):
    for action in action_histories['preflop']:
        if action["action"] == "BIGBLIND" and action["uuid"] == player_uuid:
            return True
    return False

def linear_schedule(start, end, current_step, total_steps):
    fraction = min(current_step / total_steps, 1.0)
    return start + fraction * (end - start)