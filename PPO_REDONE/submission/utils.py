# UTILS
import math
import random
from typing import Any, Final, List, Optional, Tuple
import numpy as np
from hand_strength import RANKS, SUITS, embed_with_community, street_to_num
from pypokerengine.api.emulator import Emulator
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck

ACTIONS: Final[List[str]] = ["fold", "raise", "call"]
ACTION_SPACE: Final[int] = len(ACTIONS)
NUM_PLAYERS: Final[int] = 2
PLAYER_ID_TO_POS = ["BB", "SB"]

def is_winner(game_result: dict, player_name: str) -> Tuple[bool, int, int]:
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2

def run_game(player1: BasePokerPlayer, name1: str, player2: BasePokerPlayer, name2: str, first: Optional[bool] = None, verbose: int = 0) -> dict[str, Any]:
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    if first is not None:
        if first:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
    else:
        if random.random() > 0.5:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name2, algorithm=player2)
            config.register_player(name=name1, algorithm=player1)
    return start_poker(config, verbose=verbose)

def run_n_games(player1: BasePokerPlayer, name1: str, player2: BasePokerPlayer, name2: str, n_games: int = 100) -> float:
    """Run n games and return win rate of player1 """
    wins = 0
    for i in range(n_games):
        result = run_game(player1, name1, player2, name2, first=i % 2 == 0)
        won, _, _ = is_winner(result, name1)
        if won:
            wins += 1
    return wins / n_games

def apply_action(emulator: Emulator, game_state: Any, action: str) -> Tuple[dict, dict]:
    updated_state, messages = RoundManager.apply_action(game_state, action)
    events = [emulator.create_event(message[1]["message"]) for message in messages]
    events = [e for e in events if e]
    if emulator._is_last_round(updated_state, emulator.game_rule):
        events += emulator._generate_game_result_event(updated_state)
    return updated_state, events

def get_stacks(seats: List[dict], your_uuid: str) -> Tuple[int, int]:
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

def is_big_blind(action_histories: dict, player_uuid: str) -> bool:
    for action in action_histories['preflop']:
        if action["action"] == "BIGBLIND" and action["uuid"] == player_uuid:
            return True
        else:
            return False
    return False

def linear_schedule(start: int, end: int, current_step: int, total_steps: int) -> float:
    fraction = min(current_step / total_steps, 1.0)
    return start + fraction * (end - start)

def extract_action(action_dict: dict) -> str:
    return action_dict['action']

def get_last_opponent_action(action_histories: dict, my_uuid: str) -> Tuple[Optional[str], Optional[str]]:
    for street in reversed(['preflop', 'flop', 'turn', 'river']):
        round_actions = action_histories.get(street, [])
        for action in reversed(round_actions):
            if action['uuid'] != my_uuid:
                return action, street
    return None, None

def get_aggresion_per_round(round_state: dict, player_uuid: str) -> Tuple[int, int]:
    """Get aggresion unique count per round (max 4) for player and opponent"""
    opp_raises = 0
    my_raises = 0
    opp_last_street = None
    my_last_street = None
    for street in ['preflop', 'flop', 'turn', 'river']:
        actions = round_state['action_histories'].get(street, [])
        for act in actions:
            if act['uuid'] == player_uuid:
                if act['action'] == 'raise' and my_last_street != street:
                    my_last_street = street
                    my_raises += 1
            else:
                if act['action'] == 'raise' and opp_last_street != street:
                    opp_last_street = street
                    opp_raises += 1
            if act['action'] == my_last_street and act['action'] == opp_last_street:
                break
    return opp_raises, my_raises

def is_facing_raise(round_state: dict, player_uuid: str) -> bool:
    for street in ['preflop', 'flop', 'turn', 'river']:
        actions = round_state['action_histories'].get(street, [])
        if len(actions) > 0:
            if actions[len(actions) - 1]['uuid'] != player_uuid and actions[0]['action'] == 'raise':
                return True
            break
    return False

def get_total_bettings(action_histories: dict, my_uuid: str) -> Tuple[int, int]:
    my_bet: int = 0
    opponent_bet: int = 0
    streets = ['preflop', 'flop', 'turn', 'river']

    for street in streets:
        if street not in action_histories:
            continue

        for action in action_histories[street]:
            uuid = action['uuid']
            if 'paid' in action:
                amount_paid = action['paid']
                if uuid == my_uuid:
                    my_bet += amount_paid
                else:
                    opponent_bet += amount_paid
            elif 'amount' in action and (action['action'] == 'SMALLBLIND' or action['action'] == 'BIGBLIND'):
                amount = action['amount']
                if uuid == my_uuid:
                    my_bet += amount
                else:
                    opponent_bet += amount
    return my_bet, opponent_bet

def get_initial_state() -> Tuple[Emulator, dict, dict]:
    players_info = {
        "uuid-0": { "name": "P0", "stack": 1000 },
        "uuid-1": { "name": "P1", "stack": 1000 }
    }
    emulator = Emulator()
    emulator.set_game_rule(player_num=2, max_round=500, small_blind_amount=10, ante_amount=0)
    game_state = emulator.generate_initial_game_state(players_info)
    game_state, events = emulator.start_new_round(game_state)

    return emulator, game_state, events

def get_initial_state_with_cards(cards: List[str], uuid: str):
    emulator, game_state, events = get_initial_state()
    for player in game_state["table"].seats.players:
        if player.uuid == uuid:
            holecard = gen_cards(cards)
            game_state = attach_hole_card(game_state, player.uuid, holecard)
        else:
            game_state = attach_hole_card_from_deck(game_state, player.uuid)
    return emulator, game_state, events

def make_hashable(obj):
    if isinstance(obj, np.ndarray):
        return tuple(obj.tolist())
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (tuple, set)):
        return tuple(make_hashable(x) for x in obj)
    else:
        return obj

# Stack to pot ratio
def normalize_spr(spr: float) -> float:
    return math.log(1 + spr) / math.log(1 + 66.3333)

def card_to_int(card_str: str) -> int:
    """Returns unique int for the card"""
    # Handle two-character and three-character card strings
    if len(card_str) == 2:
        suit = SUITS.index(card_str[0])
        rank = RANKS.index(card_str[1])
    elif len(card_str) == 3:
        suit = SUITS.index(card_str[0])
        rank = int(card_str[1:])
    else:
        raise ValueError(f"Invalid card string: {card_str}")

def convert_hand(card_strings: List[str], max_len: int = 2) -> np.ndarray:
    hand = [card_to_int(card) for card in card_strings]
    padded_hand = hand + [-1] * (max_len - len(hand))  # Pad with -1
    return np.array(padded_hand[:max_len], dtype=np.int32)

def encode_infoset_alt(round_state: dict, hole_cards: List[str], community_cards: List[str], player_uuid: str) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    position = 1 if is_big_blind(round_state['action_histories'], player_uuid) else 0
    agg, agg_opp = get_aggresion_per_round(round_state, player_uuid)
    street_idx = street_to_num(round_state['street'])
    stack, _ = get_stacks(round_state['seats'], player_uuid)
    spr = normalize_spr(stack / round_state['pot']['main']['amount'])
    facing_raise = 1 if is_facing_raise(round_state, player_uuid) else 0
    bettings = np.array([
        position,
        spr,
        facing_raise,
        agg / 4,
        agg_opp / 4,
    ], dtype=np.float32)

    return (
        convert_hand(hole_cards, 2),
        convert_hand(community_cards, 5),
        np.int32(street_idx),
        bettings
    )


def encode_infoset(round_state: dict, hole_cards: List[str], community_cards: List[str], player_uuid: str) -> np.ndarray:
    position = 1 if is_big_blind(round_state['action_histories'], player_uuid) else 0
    agg, agg_opp = get_aggresion_per_round(round_state, player_uuid)
    street_idx = street_to_num(round_state['street'])
    stack, _ = get_stacks(round_state['seats'], player_uuid)
    spr = normalize_spr(stack / round_state['pot']['main']['amount'])
    facing_raise = 1 if is_facing_raise(round_state, player_uuid) else 0
    hand = embed_with_community(hole_cards=hole_cards, community_cards=community_cards)
    all = np.array(hand + [
        street_idx,
        position,
        spr,
        facing_raise,
        agg / 4,
        agg_opp / 4,
    ], dtype=np.float32)
    return all

def get_payoff(round_state: dict, player_idx: int) -> float:
    is_winner = round_state['seats'][0 if player_idx == 1 else 0]['state'] == 'folded'
    paid, opp_paid = get_total_bettings(round_state['action_histories'], round_state['seats'][player_idx]['uuid'])
    return (opp_paid if is_winner else -paid) / 1000.