import torch
from cfr import get_info_set
from cfr_model import DeepCFRModel
from fast_hand_strength import estimate_hand_strength
from pypokerengine.players import BasePokerPlayer
from utils import ACTIONS,get_device
import random as rand

class CFRPokerPlayer(BasePokerPlayer):
    def __init__(self, model: DeepCFRModel):
        super().__init__()
        self.device = get_device()
        self.model: DeepCFRModel = model
        self.model.to(self.device)
        self.model.eval()

    def declare_action(self, valid_actions, hole_card, round_state):
        with torch.no_grad():
            hole_cards, community_cards, street, betting_feats = get_info_set(round_state, hole_card, round_state['community_card'], self.uuid)
            logits = self.model(
                torch.tensor(hole_cards, device=self.device).unsqueeze(0),
                torch.tensor(community_cards, device=self.device).unsqueeze(0),
                torch.tensor(street, device=self.device).unsqueeze(0),
                torch.tensor(betting_feats, device=self.device).unsqueeze(0)
            ).squeeze(0)
            mask = torch.tensor(
                [action in valid_actions for action in ACTIONS],
                device=logits.device,
                dtype=torch.bool
            )
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            chosen_idx = torch.argmax(masked_logits).item()
            return ACTIONS[chosen_idx]

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
            simulations=max(int(100 * self.difficulty), 10)
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

class CallPlayer(BasePokerPlayer):
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
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class BluffPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
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
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class FoldPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        return valid_actions[0]["action"]
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass