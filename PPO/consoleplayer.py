# Play using console
import pprint
pp = pprint.PrettyPrinter()
from ppoplayer import PPBomb
from utils import run_game
from pypokerengine import BasePokerPlayer 


class ConsolePlayer(BasePokerPlayer):
  def __init__(self, input_receiver=None):
    self.input_receiver = input_receiver if input_receiver else self.__gen_raw_input_wrapper()

  def declare_action(self, valid_actions, hole_card, round_state):
    print('========= DECLARE ACTION =========')
    pp.pprint(round_state)
    pp.pprint({
        'valid_actions': valid_actions,
        'hole_card': hole_card,
        'community_card': round_state['community_card'],
    })
    action = self.__receive_action_from_console(valid_actions)
    return action

  def receive_game_start_message(self, game_info):
    print('\n========= NEW GAME =========')
    pp.pprint(game_info)
    self.__wait_until_input('New game started')

  def receive_round_start_message(self, round_count, hole_card, seats):
    print('\n\n========= ROUND START =========')
    pp.pprint({
        'round_count': round_count,
        'hole_card': hole_card,
        'seats': seats
    })
    #self.__wait_until_input()

  def receive_street_start_message(self, street, round_state):
    print('========= STREET START =========')
    pp.pprint(street)
    #self.__wait_until_input()

  def receive_game_update_message(self, new_action, round_state):
    print('========= GAME UPDATE =========')
    pp.pprint(new_action)
    #self.__wait_until_input()

  def receive_round_result_message(self, winners, hand_info, round_state):
    print('========= ROUND RESULT =========')
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