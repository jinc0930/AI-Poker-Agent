from utils import get_stacks
from pypokerengine.players import BasePokerPlayer
from model import PPO
import torch
from torch.distributions import Categorical
from encoder import encode
from pypokerengine.players import BasePokerPlayer

class PPBomb(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if not hasattr(self, 'model'):
            self.model = PPO(filename='/content/star2/Star2_23502.pt')
            self.model.eval()
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        call_action = next((a for a in valid_actions if a['action'] == 'call'), None)
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
            player_id=self.uuid,
            amount_to_call=call_action.get('amount', 0)
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
    return PPBomb()