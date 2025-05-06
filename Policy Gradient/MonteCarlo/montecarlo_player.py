from pypokerengine.players import BasePokerPlayer
from .hand_strenth import estimate_hand_strength

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