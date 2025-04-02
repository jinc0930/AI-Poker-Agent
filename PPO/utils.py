from model import Hyperparams, PPO
from encoder import calculate_bets, encode
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import torch
from torch.distributions import Categorical
import random as rand
import random
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
    def __init__(self, filename=None, hyperparams=Hyperparams(), disable_training=False):
        super().__init__()
        self.model = PPO(filename=filename, hyperparams=hyperparams)
        self.disable_training = disable_training
        self.folded_with_reward = False
        self.prev_wr = None
        self.prev_hole_card = None
        self.prev_community_card = None

        # stats
        self.hands = 0
        self.folds = 0
        self.raises = 0
        self.calls = 0
        self.rewards = 0
        self.rewards_from_folding = 0

        # track
        self.last_wr = None
        self.last_hole_card = None
        self.last_community_card = None


    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        # call_action = next((a for a in valid_actions if a['action'] == 'call'), None)
        if not hasattr(self, 'memory'):
            self.memory = [round_state['action_histories']]
            self.last_round_memory = round_state['round_count']
        else:
            if self.last_round_memory == round_state['round_count']:
                self.memory[-1] = round_state['action_histories']
            else:
                self.memory.append(round_state['action_histories'])
                self.last_round_memory = round_state['round_count']

        is_sb = int(any(action['action'] == 'SMALLBLIND' and action['uuid'] == self.uuid for action in round_state['action_histories']['preflop']))
        state_prime = encode(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            street=round_state["street"],
            pot_size=round_state["pot"]["main"]["amount"],
            stack=my_stack,
            opponent_stack=opponent_stack,
            round_count=round_state["round_count"],
            is_small_blind = is_sb,
            all_action_histories=self.memory,
            player_id=self.uuid,
        )

        if not hasattr(self, 'state'):
            self.state = state_prime

        prob = self.model.pi(torch.tensor(self.state))
        action = Categorical(prob).sample().item()

        # Training part
        if not self.disable_training:
            reward = 0
            self.model.put_data((self.state, action, reward, state_prime, prob[action].item(), False))
        self.state = state_prime

        if action == 0:
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    self.raises += 1
                    return action_info["action"]
            # fallback to call
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
            paid1, paid2 = calculate_bets(round_state['action_histories'], self.uuid)
            if winners[0]['uuid'] == self.uuid:
                reward = (paid2 / 2000) * (self.hands / 500)
            else:
                reward = (paid1 / -2000) * (self.hands / 500)
            self.rewards += reward
            self.model.reward(reward)
    def done_with_reward(self, reward):
        if not self.disable_training:
            self.model.reward(reward)
            self.rewards += reward
            self.model.done()
            self.model.train_net(force = True)

def run_game(player1, name1, player2, name2, first = True, verbose = 0):
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    if random.random() > 0.5:
        config.register_player(name=name1, algorithm=player1)
        config.register_player(name=name2, algorithm=player2)
    else:
        config.register_player(name=name2, algorithm=player2)
        config.register_player(name=name1, algorithm=player1)
    return start_poker(config, verbose=verbose)

def is_winner(game_result: Dict, player_name: str):
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2