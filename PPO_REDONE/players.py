import torch
from model import PPO, Hyperparams
from utils import encode_infoset, get_total_bettings
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
# from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, Card, gen_cards
from torch.distributions import Categorical
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

class AITrainer(BasePokerPlayer):
    def __init__(self, filename=None, hyperparams=Hyperparams(), disable_training=False, is_opponent=False):
        super().__init__()
        self.model = PPO(filename=filename, hyperparams=hyperparams)
        self.disable_training = disable_training
        self.folded_with_reward = False
        self.is_opponent = is_opponent

        # Stats
        self.hands = 0
        self.folds = 0
        self.raises = 0
        self.calls = 0
        self.rewards = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        state_prime = encode_infoset(round_state, hole_card, round_state['community_card'], self.uuid)

        if not hasattr(self, 'state'):
            self.state = state_prime

        if self.disable_training:
            self.model.eval()

        if self.is_opponent:
            action = self.model.get_opponent_action(torch.tensor(self.state))
        else:
            prob = self.model.pi(torch.tensor(self.state))
            action = Categorical(prob).sample().item()

        # Training part
        if not self.disable_training:
            reward = 0
            # if action == 2:
            #     win_rate, loss_rate, tie_rate = monte_carlo(hole_card, round_state["community_card"], samples=100)
            #     if win_rate <= 0.5:
            #         reward = 0
            #         self.folded_with_reward = True
            self.model.put_data((self.state, action, reward, state_prime, prob[action].item(), False))
        self.state = state_prime
        if action == 0:
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    self.raises += 1
                    return action_info["action"]
            # Fallback to call
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        elif action == 1:
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    self.calls += 1
                    return action_info["action"]
        else:
          for action_info in valid_actions:
                if action_info["action"] == "fold":
                    self.folds += 1
                    return action_info["action"]
        return valid_actions[0]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1
        reward = 0
        if self.folded_with_reward == True:
            self.folded_with_reward = False
        else:
            paid1, paid2 = get_total_bettings(round_state['action_histories'], self.uuid)
            if winners[0]['uuid'] == self.uuid:
                reward = (paid2 / 1000) * (self.hands / 500)
            else:
                reward = (paid1 / -1000) * (self.hands / 500)
            self.rewards += reward
            self.model.reward(reward)
    def done_with_reward(self, reward):
        if not self.disable_training:
            self.model.reward(reward)
            self.rewards += reward
            self.model.done()
            self.model.train_net(force = True)


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

