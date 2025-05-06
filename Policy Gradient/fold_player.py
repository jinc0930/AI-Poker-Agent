from pypokerengine.players import BasePokerPlayer

class FoldOnlyPlayer(BasePokerPlayer):
    """A player that always folds unless checking is free."""
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # Check if we can check (call with 0 amount)
        for action in valid_actions:
            if action['action'] == 'call' and action.get('amount', 0) == 0:
                return 'call'  # This is actually a check
        
        # Otherwise fold
        return 'fold'
        
    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return FoldOnlyPlayer()