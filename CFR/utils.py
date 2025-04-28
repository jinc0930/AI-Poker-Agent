
import random
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from pypokerengine.api.emulator import Emulator
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import attach_hole_card, attach_hole_card_from_deck

ACTIONS = ['fold', 'raise', 'call']

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
    for i in tqdm(range(n_games)):
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
    
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")