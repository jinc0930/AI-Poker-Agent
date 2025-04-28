suits = ['d','s','c','h']
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
cards = []
for r in ranks:
    for s in suits:
        cards.append(r+s)

def get_stacks(seats, your_uuid):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

def is_big_blind(action_histories, player_uuid):
    for action in action_histories['preflop']:
        if action["action"] == "BIGBLIND" and action["uuid"] == player_uuid:
            return True
    return False

def linear_schedule(start, end, current_step, total_steps):
    fraction = min(current_step / total_steps, 1.0)
    return start + fraction * (end - start)