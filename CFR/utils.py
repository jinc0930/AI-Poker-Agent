import random
from typing import Any, List, Optional, Tuple
from tqdm import tqdm
from pypokerengine.api.emulator import Emulator
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.engine.round_manager import RoundManager
from pypokerengine.players import BasePokerPlayer

def is_winner(game_result: dict, player_name: str):
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2

def run_game(player1: BasePokerPlayer, name1: str, player2: BasePokerPlayer, name2: str, first: Optional[bool] = None, verbose: int = 0):
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

def run_n_games(player1: BasePokerPlayer, name1: str, player2: BasePokerPlayer, name2: str, n_games: int = 100):
    """Run n games and return win rate of player1 """
    wins = 0
    for i in tqdm(range(n_games)):
        result = run_game(player1, name1, player2, name2, first=i % 2 == 0)
        won, _, _ = is_winner(result, name1)
        if won:
            wins += 1
    return wins / n_games

def apply_action(emulator: Emulator, game_state: Any, action: str):
    updated_state, messages = RoundManager.apply_action(game_state, action)
    events = [emulator.create_event(message[1]["message"]) for message in messages]
    events = [e for e in events if e]
    if emulator._is_last_round(updated_state, emulator.game_rule):
        events += emulator._generate_game_result_event(updated_state)
    return updated_state, events

def get_stacks(seats: List[dict], your_uuid: str):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

def is_big_blind(action_histories: dict, player_uuid: str):
    for action in action_histories['preflop']:
        if action["action"] == "BIGBLIND" and action["uuid"] == player_uuid:
            return True
        else:
            return False
    return False

def linear_schedule(start: int, end: int, current_step: int, total_steps: int):
    fraction = min(current_step / total_steps, 1.0)
    return start + fraction * (end - start)

def extract_action(action_dict: dict):
    return action_dict['action']

def get_last_opponent_action(action_histories: dict, my_uuid: str):
    for street in reversed(['preflop', 'flop', 'turn', 'river']):
        round_actions = action_histories.get(street, [])
        for action in reversed(round_actions):
            if action['uuid'] != my_uuid:
                return action, street
    return None, None

def get_aggresion_freq(round_state: dict, player_uuid: str) -> Tuple[float, float]:
    """Get aggresion frequency for player and opponent"""
    opp_raises = opp_actions = 0
    my_raises = my_actions = 0
    for street in ['preflop', 'flop', 'turn', 'river']:
        actions = round_state['action_histories'].get(street, [])
        for act in actions:
            if act['uuid'] == player_uuid:
                my_actions += 1
                if act['action'] == 'raise': my_raises += 1
            else:
                opp_actions += 1
                if act['action'] == 'raise': opp_raises += 1

    opp_aggr = opp_raises / opp_actions if opp_actions else 0.0
    my_aggr = my_raises / my_actions if my_actions else 0.0
    return (my_aggr, opp_aggr)