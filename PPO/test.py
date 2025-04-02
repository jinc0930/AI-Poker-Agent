# Play using console
from ppoplayer import PPBomb
from utils import run_game
from pypokerengine.players import BasePokerPlayer
import pypokerengine.utils.visualize_utils as U


class ConsolePlayer(BasePokerPlayer):
  def __init__(self, input_receiver=None):
    self.input_receiver = input_receiver if input_receiver else self.__gen_raw_input_wrapper()

  def declare_action(self, valid_actions, hole_card, round_state):
    print(U.visualize_declare_action(valid_actions, hole_card, round_state, self.uuid))
    action, amount = self.__receive_action_from_console(valid_actions)
    return action, amount

  def receive_game_start_message(self, game_info):
    print(U.visualize_game_start(game_info, self.uuid))
    self.__wait_until_input('New game started')

  def receive_round_start_message(self, round_count, hole_card, seats):
    print(U.visualize_round_start(round_count, hole_card, seats, self.uuid))
    #self.__wait_until_input()

  def receive_street_start_message(self, street, round_state):
    print(U.visualize_street_start(street, round_state, self.uuid))
    #self.__wait_until_input()

  def receive_game_update_message(self, new_action, round_state):
    print(U.visualize_game_update(new_action, round_state, self.uuid))
    #self.__wait_until_input()

  def receive_round_result_message(self, winners, hand_info, round_state):
    print(U.visualize_round_result(winners, hand_info, round_state, self.uuid))
    self.__wait_until_input(f"Round finished! (Winner: {winners[0]['name']})")

  def __wait_until_input(self, msg = ''):
    input(f'{msg} Enter some key to continue ...')

  def __gen_raw_input_wrapper(self):
    return lambda msg: input(msg)

  def __receive_action_from_console(self, valid_actions):
    flg = self.input_receiver('Enter f(fold), c(call), r(raise).\n >> ')
    if flg in self.__gen_valid_flg(valid_actions):
      if flg == 'f':
        return valid_actions[0]['action']
      elif flg == 'c':
        return valid_actions[1]['action']
      elif flg == 'r':
        return valid_actions[2]['action']
    else:
      return self.__receive_action_from_console(valid_actions)

  def __gen_valid_flg(self, valid_actions):
    flgs = ['f', 'c']
    is_raise_possible = len(valid_actions) >= 3
    if is_raise_possible:
      flgs.append('r')
    return flgs

if __name__ == '__main__':
    run_game(ConsolePlayer(), 'You', PPBomb(), 'PPBomb')