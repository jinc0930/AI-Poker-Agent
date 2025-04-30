from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random as rand
from typing import Dict

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

class CallPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [FOLD, CALL, RAISE]
        for action_info in valid_actions:
            if action_info["action"] == "call":
                return action_info["action"]
        return valid_actions[0]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class BluffPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, _ = get_stacks(round_state["seats"], self.uuid)
        # valid_actions format => [FOLD, CALL, RAISE]
        for action_info in valid_actions:
            if action_info["action"] == "raise":
                return action_info["action"]
        for action_info in valid_actions:
            if action_info["action"] == "call":
                return action_info["action"]
        return valid_actions[0]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0/3

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [ 1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def declare_action(self, valid_actions, hole_card, round_state):
        action = self.__choice_action(valid_actions)
        if action == "raise":
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    return action_info["action"]
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        elif action == "call":
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        else:
            for action_info in valid_actions:
                if action_info["action"] == "fold":
                    return action_info["action"]
        return valid_actions[0]["action"]

    def __choice_action(self, valid_actions):
        r = rand.random()
        if r <= self.fold_ratio:
            return 'fold'
        elif r <= self.call_ratio:
            return 'call'
        return 'raise'

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

def is_winner(game_result: Dict, player_name: str):
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2

def run_game(player1, name1, player2, name2, first = None, verbose = 0):
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    if first is not None:
        if first:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
    else:
        if rand.random() > 0.5:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name2, algorithm=player2)
            config.register_player(name=name1, algorithm=player1)
    return start_poker(config, verbose=verbose)

