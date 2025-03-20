from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random as rand
from model import PPO 
from randomplayer import RandomPlayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from encoder import card_to_num


class FishPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
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

class AITrainer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        cards = [card_to_num(hole_card[0]), card_to_num(hole_card[1])]
        card_mask = [False, False]
        for i in range(0, 5):
            if i < len(round_state['community_card']):
                cards.append(card_to_num(round_state['community_card'][i]))
                card_mask.append(False)
            else:
                cards.append([-1, -1])
                card_mask.append(True)
        cards = torch.tensor([cards], dtype=torch.long)
        card_mask = torch.tensor([card_mask], dtype=torch.bool)
        card_features = self.model.encode_cards(cards, card_mask)
        additional_features = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        combined_features = torch.cat([card_features, additional_features], dim=1)

        prob = self.model.pi(combined_features)
        action = Categorical(prob).sample().item()
        reward = 0
        self.model.put_data((combined_features, action, reward, combined_features, prob[0][action].item(), False))

        # valid_actions format => [FOLD, CALL, RAISE]
        final = valid_actions[action]
        if action == 2:
            return final["action"], final["amount"]["min"] + 10
        return final["action"], final["amount"]

    def receive_game_start_message(self, game_info):
        if (hasattr(self, 'agent')):
            self.games += 1
            return
        else:
            self.model = PPO()
            self.games = 1

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if winners[0]['name'] == "AIPlayer":
            self.model.reward()
        if (self.games % 20) == 0:
            self.model.update()
        pass


if __name__ == "__main__":
    iterations = 10_000

    ai = AITrainer()
    config = setup_config(max_round=100, initial_stack=1000, small_blind_amount=10)
    config.register_player(name="AIPlayer", algorithm=ai)
    config.register_player(name="RandomPlayer", algorithm=RandomPlayer())

    def determine_winner(game_data):
        players = game_data['players']
        winner = max(players, key=lambda player: player['stack'])
        return winner['name']

    games_data = []
    for i in range(iterations):
        # verbose = 1 to show trace
        game_result = start_poker(config, verbose=0)
        games_data.append(game_result)

    stack = 0
    wins = 0
    games = len(games_data)

    for game_data in games_data:
        players = game_data['players']
        # Find the winner in this game
        winner = max(players, key=lambda player: player['stack'])

        # Update PPOTrainer's stack and win count
        if winner['name'] == 'AIPlayer':
            wins += 1
            stack += winner['stack']
        else:
            # If PPOTrainer didn't win, just add their stack to the total
            stack += next(player['stack'] for player in players if player['name'] == 'AIPlayer')

    print(f"PPOTrainer's games: {games}")
    print(f"PPOTrainer's final stack: {stack}")
    print(f"PPOTrainer's total wins: {wins}")
    game_result = start_poker(config, verbose=1)