import torch
from feature_extraction import encode
from pypokerengine.api.game import setup_config, start_poker
from utils import BluffPlayer, RandomPlayer, get_stacks, is_winner, run_game
from model import PPO
from pypokerengine.players import BasePokerPlayer
from torch.distributions import Categorical

class CustomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if not hasattr(self, 'model'):
            self.model = PPO(filename='./Star_110100.pt')
            self.model.eval()
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        if not hasattr(self, 'memory'):
            self.memory = [round_state['action_histories']]
            self.last_round_memory = round_state['round_count']
        else:
            if self.last_round_memory == round_state['round_count']:
                self.memory[-1] = round_state['action_histories']
            else:
                self.memory.append(round_state['action_histories'])
                self.last_round_memory = round_state['round_count']

        state_prime = encode(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            street=round_state["street"],
            pot_size=round_state["pot"]["main"]["amount"],
            stack=my_stack,
            opponent_stack=opponent_stack,
            round_count=round_state["round_count"],
            is_small_blind = int(any(action['action'] == 'SMALLBLIND' and action['uuid'] == self.uuid for action in round_state['action_histories'].get('preflop', []))),
            all_action_histories=self.memory,
            player_id=self.uuid
        )

        if not hasattr(self, 'state'):
            self.state = state_prime

        prob = self.model.pi(torch.tensor(self.state))
        action = Categorical(prob).sample().item()
        self.state = state_prime

        if action == 0:
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    return action_info["action"]
            # fallback to call
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        elif action == 1:
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        else:
          for action_info in valid_actions:
                if action_info["action"] == "fold":
                    return action_info["action"]
        return valid_actions[0]["action"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return CustomPlayer()









# test

def evaluate(episodes = 100):
    wins = 0
    chips = 0
    for i in range(episodes):
        p1 = CustomPlayer()
        p2 = BluffPlayer()
        with torch.no_grad():
            game_result = run_game(p1, 'CustomPlayer', p2, 'BluffPlayer', first=i % 2 == 0)
            is_win, chips, _ = is_winner(game_result, 'CustomPlayer')
            wins += 1 if is_win else 0
            chips += (chips - 1000)
    return wins / episodes, chips / episodes

if __name__ == '__main__':
    wr, chips = evaluate()
    print(f'yippie, wr = {wr}')