from pypokerengine.players import BasePokerPlayer
from model import PPO
import torch
from torch.distributions import Categorical
from encoder import encode
from pypokerengine.players import BasePokerPlayer

def get_stacks(seats, your_uuid):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

# Player
class PPBomb(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if not hasattr(self, 'model'):
            self.model = PPO(filename='./PPBomb.pt')
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        state_prime = encode(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            street=round_state["street"],
            pot_size=round_state["pot"]["main"]["amount"],
            stack=my_stack,
            opponent_stack=opponent_stack,
            round_count=round_state["round_count"],
            is_small_blind = int(any(action['action'] == 'SMALLBLIND' and action['uuid'] == self.uuid for action in round_state['action_histories'].get('preflop', []))),
            action_histories=round_state['action_histories'],
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
                    return action_info["action"], action_info["amount"]["min"]
        elif action == 1:
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"], action_info["amount"]
        else:
          for action_info in valid_actions:
                if action_info["action"] == "fold":
                    return action_info["action"], action_info["amount"]
        return valid_actions[0]["action"], valid_actions[0]["amount"]

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return PPBomb()