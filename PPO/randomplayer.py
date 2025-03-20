from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):
  def declare_action(self, valid_actions, hole_card, round_state):
    r = rand.random()
    if r <= 0.33:
      action = 2 # raise
    if r <= 0.66:
      action = 1 # call 
    else:
      action = 0 # fold
      
    final = valid_actions[action]
    if action == 2:
        return final["action"], final["amount"]["min"] + 10
    return final["action"], final["amount"]

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

def setup_ai():
  return RandomPlayer()