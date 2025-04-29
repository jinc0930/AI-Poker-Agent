import torch
from utils import encode_infoset
from model import PPO
from pypokerengine.players import BasePokerPlayer
from torch.distributions import Categorical

class CustomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if not hasattr(self, 'model'):
            self.model = PPO(filename='./FrozenStar_40100.pt')
            self.model.eval()
        state_prime = encode_infoset(round_state, hole_card, round_state['community_card'], self.uuid)

        if not hasattr(self, 'state'):
            self.state = state_prime

        prob = self.model.pi(torch.tensor(self.state))
        action = Categorical(prob).sample().item()

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
    def receive_round_result_message(self, winners, hand_info, round_state): pass