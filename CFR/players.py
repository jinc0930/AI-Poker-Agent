import numpy as np
from cfr import  CFRTree, get_info_state
from hand_strenth import estimate_hand_strength
from pypokerengine.engine.card import Card
from pypokerengine.players import BasePokerPlayer
from utils import extract_action
import random as rand

class CFRPokerPlayer(BasePokerPlayer):
    def __init__(self, game_tree: CFRTree):
        super().__init__()
        self.game_tree = game_tree

    def declare_action(self, valid_actions, hole_card, round_state):
        info_state = get_info_state(round_state, hole_card, round_state['community_card'], self.uuid)
        node, _ = self.game_tree.get_nearest_node(round_state['street'], info_state)
        strategy = node.get_average_strategy()
        valid_strategy = list(map(extract_action, valid_actions))
        probs = [strategy.get(a, 0.0) for a in valid_strategy]
        prob_sum = sum(probs)
        if prob_sum <= 0:
            return "call"

        best_action = valid_strategy[np.argmax(probs)]
        return best_action

    @classmethod
    def load_from_file(cls, filename = './cfr.pickle'):
        game_tree = CFRTree.load(filename)
        return cls(game_tree)

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, difficulty=1):
        super().__init__()
        self.difficulty = max(0.0, min(1.0, difficulty))  # Clamp to [0,1]
        self.hands = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        win_rate, _, _ = estimate_hand_strength(
            hole_cards = hole_card,
            community_cards = round_state["community_card"],
            samples=max(int(100 * self.difficulty), 10)
        )
        raise_action = next((a for a in valid_actions if a['action'] == 'raise'), None)
        actions = (1 if action['action'] == 'RAISE' else 0 for street in reversed(['preflop', 'flop', 'turn', 'river'])
                for action in reversed(round_state['action_histories'].get(street, [])) if action['uuid'] != self.uuid)
        is_raise = next(actions, 0)
        raise_threshold = 0.3 + (0.5 * self.difficulty)
        call_threshold = 0.2 + (0.4 * self.difficulty)
        action = 'fold'

        if is_raise == 0:  # Check situation
            if win_rate >= raise_threshold and raise_action:
                action = 'raise'
            else:
                action = 'call'  # Check
        else:  # Facing a bet
            if win_rate >= raise_threshold and raise_action:
                action = 'raise'
            elif win_rate >= call_threshold:
                action = 'call'
            else:
                action = 'fold'

        return action
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        r = rand.random()
        if r <= 0.5:
            call_action_info = valid_actions[1]
        elif r<= 0.9 and len(valid_actions) == 3:
          call_action_info = valid_actions[2]
        else:
           call_action_info = valid_actions[0]
        action = call_action_info["action"]
        return action

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass