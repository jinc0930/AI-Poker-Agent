from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random as rand
import numpy as np
from model import PPO, T_horizon
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from encoder import card_to_num, encode
import os

class FishPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [FOLD, CALL, RAISE]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [FOLD, CALL, RAISE]
        r = rand.random()
        if r < 0.33:
            action, amount = valid_actions[0]["action"], valid_actions[0]["amount"]
        elif r < 0.66:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            action, amount = valid_actions[2]["action"], valid_actions[2]["amount"]["min"] + 10
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def get_stacks(seats, your_uuid):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

class AITrainer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        state_prime = encode(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            street=round_state["street"],
            pot_size=round_state["pot"]["main"]["amount"],
            stack=my_stack,
            opponent_stack=opponent_stack,
            round_count=round_state["round_count"],
            is_small_blind=int(any(action['action'] == 'SMALLBLIND' and action['uuid'] == self.uuid for action in round_state['action_histories'].get('preflop', [])))
        )

        if not hasattr(self, 'state'):
            self.state = state_prime

        prob = self.model.pi(torch.tensor(self.state, dtype=torch.float))
        action = Categorical(prob).sample().item()
        reward = 0
        self.model.put_data((self.state, action, reward, state_prime, prob[action].item(), False))
        self.state = state_prime
        self.actions += 1

        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        final = valid_actions[action]
        if action == 2:
            return final["action"], final["amount"]["min"] + 10
        return final["action"], final["amount"]

    def receive_game_start_message(self, game_info):
        if not hasattr(self, 'model'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = PPO()
        self.actions = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if winners[0]['uuid'] == self.uuid:
            self.model.reward(round_state['pot']['main']['amount'] / 2000)
        else:
            self.model.reward(round_state['pot']['main']['amount'] / -2000)
        if (self.actions % T_horizon) == 0:
            self.model.train_net()
        pass

def run_game(player1, name1, player2, name2, verbose = 0):
    config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=10)
    config.register_player(name=name1, algorithm=player1)
    config.register_player(name=name2, algorithm=player2)
    return start_poker(config, verbose=verbose)

if __name__ == '__main__':
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='PPO/runs/nowyes_150k')

    def determine_winner(game_data):
        players = game_data['players']
        winner = max(players, key=lambda player: player['stack'])
        return winner['name']

    stack = 0
    wins = 0
    iterations = 10000

    ai = AITrainer()
    opponent = FishPlayer()

    for i in range(iterations):
        toggle = i % 2 == 0
        player1 = ai if toggle else opponent
        player2 = opponent if toggle else ai
        name1 = 'AIPlayer' if toggle else 'RandomPlayer'
        name2 = 'RandomPlayer' if toggle else 'AIPlayer'
        game_result = run_game(player1, name1, player2, name2)

        # Process current game
        players = game_result['players']
        winner = max(players, key=lambda player: player['stack'])
        ai_stack = next(player['stack'] for player in players if player['name'] == 'AIPlayer')

        stack += ai_stack
        if winner['name'] == 'AIPlayer':
            wins += 1

        writer.add_scalar('AIPlayer/Stack', ai_stack, i)
        writer.add_scalar('AIPlayer/Cumulative_Stack', stack, i)
        writer.add_scalar('AIPlayer/Wins', wins, i)
        writer.add_scalar('AIPlayer/Win_Rate', wins/(i + 1), i)

    # Close the writer
    writer.close()