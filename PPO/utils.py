from phevaluator import evaluate_cards
import random
from typing import List
import numpy as np
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
# from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, Card, gen_cards
from torch.utils.tensorboard import SummaryWriter
import random as rand
from typing import Dict
from model import Hyperparams, PPO
from encoder import calculate_bets, encode
import torch
from torch.distributions import Categorical
import random as rand
import random
from typing import Dict

suits = ['d','s','c','h']
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
cards = []
for r in ranks:
    for s in suits:
        cards.append(r+s)

def simulate(hand: List[str], table:List[str], players = 2):
    hands = []
    deck = random.sample(cards,len(cards))
    hand = [card[1] + card[0].lower() for card in hand]
    table = [card[1] + card[0].lower() for card in table]
    full = table + hand
    deck = list(filter(lambda x: x not in full, deck))
    for i in range(players):
        hn = []
        hn.append(deck[0])
        deck = deck[1:]
        hn.append(deck[0])
        deck = deck[1:]
        hands.append(hn)
    while len(table) < 5:
        card = deck.pop(0)
        table.append(card)
        full.append(card)
    my_hand_rank = evaluate_cards(full[0],full[1],full[2],full[3],full[4],full[5],full[6])

    for check_hand in hands:
        all_cards = table + check_hand
        opponent = evaluate_cards(all_cards[0],all_cards[1],all_cards[2],all_cards[3],all_cards[4],all_cards[5],all_cards[6])
        if opponent < my_hand_rank:
            return 1
        if opponent == my_hand_rank:
            return 2
        return 0

def monte_carlo(hand, table, players=2, samples: int =100):
    outcomes = np.zeros(3, dtype=np.int32)
    for _ in range(samples):
        outcome = simulate(hand, table, players)
        outcomes[outcome] += 1
    return outcomes / samples


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


class MonteCarloPlayer(BasePokerPlayer):
    def __init__(self, difficulty=0.5):
        super().__init__()
        self.difficulty = max(0.0, min(1.0, difficulty))  # Clamp to [0,1]
        self.hands = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        win_rate, loss_rate, tie_rate = monte_carlo(hole_card, round_state["community_card"], samples=max(int(100 * self.difficulty), 10))

        # Find valid actions
        raise_action = next((a for a in valid_actions if a['action'] == 'raise'), None)
        call_action = next((a for a in valid_actions if a['action'] == 'call'), None)
        fold_action = next((a for a in valid_actions if a['action'] == 'fold'), None)

        # Check if we're facing a raise
        actions = (1 if action['action'] == 'RAISE' else 0 for street in reversed(['preflop', 'flop', 'turn', 'river'])
                for action in reversed(round_state['action_histories'].get(street, [])) if action['uuid'] != self.uuid)
        is_raise = next(actions, 0)

        # Calculate thresholds based on difficulty
        # Higher difficulty = higher thresholds = tighter play
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
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class CallPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
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
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class BluffPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, _ = get_stacks(round_state["seats"], self.uuid)
        # valid_actions format => [FOLD, CALL, RAISE]
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
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class RandomPlayer(BasePokerPlayer):
    def __init__(self):
        super().__init__()
        self.hands = 0
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0/3

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [ 1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def declare_action(self, valid_actions, hole_card, round_state):
        action = self.__choice_action(valid_actions)
        if action == "raise":
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    return action_info["action"]
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        elif action == "call":
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"]
        else:
            for action_info in valid_actions:
                if action_info["action"] == "fold":
                    return action_info["action"]
        return valid_actions[0]["action"]

    def __choice_action(self, valid_actions):
        r = rand.random()
        if r <= self.fold_ratio:
            return 'fold'
        elif r <= self.call_ratio:
            return 'call'
        return 'raise'

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class AITrainer(BasePokerPlayer):
    def __init__(self, filename=None, hyperparams=Hyperparams(), disable_training=False, is_opponent=False):
        super().__init__()
        self.model = PPO(filename=filename, hyperparams=hyperparams)
        self.disable_training = disable_training
        self.folded_with_reward = False
        self.is_opponent = is_opponent

        # stats
        self.hands = 0
        self.folds = 0
        self.raises = 0
        self.calls = 0
        self.rewards = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        my_stack, opponent_stack = get_stacks(round_state["seats"], self.uuid)
        # call_action = next((a for a in valid_actions if a['action'] == 'call'), None)
        if not hasattr(self, 'memory'):
            self.memory = [round_state['action_histories']]
            self.last_round_memory = round_state['round_count']
        else:
            if self.last_round_memory == round_state['round_count']:
                self.memory[-1] = round_state['action_histories']
            else:
                self.memory.append(round_state['action_histories'])
                self.last_round_memory = round_state['round_count']

        is_sb = int(any(action['action'] == 'SMALLBLIND' and action['uuid'] == self.uuid for action in round_state['action_histories']['preflop']))
        state_prime = encode(
            hole_cards=hole_card,
            community_cards=round_state["community_card"],
            street=round_state["street"],
            pot_size=round_state["pot"]["main"]["amount"],
            stack=my_stack,
            opponent_stack=opponent_stack,
            round_count=round_state["round_count"],
            is_small_blind = is_sb,
            all_action_histories=self.memory,
            player_id=self.uuid,
        )

        if not hasattr(self, 'state'):
            self.state = state_prime

        if self.disable_training:
            self.model.eval()

        if self.is_opponent:
            action = self.model.get_opponent_action(torch.tensor(self.state))
        else:
            prob = self.model.pi(torch.tensor(self.state))
            action = Categorical(prob).sample().item()

        # Training part
        if not self.disable_training:
            reward = 0
            if action == 2:
                win_rate, loss_rate, tie_rate = monte_carlo(hole_card, round_state["community_card"], samples=100)
                if win_rate <= 0.5:
                    reward = 0
                    self.folded_with_reward = True

            self.model.put_data((self.state, action, reward, state_prime, prob[action].item(), False))
        self.state = state_prime
        if action == 0:
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    self.raises += 1
                    return action_info["action"]
            # fallback to call
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
    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1
        reward = 0
        if self.folded_with_reward == True:
            self.folded_with_reward = False
        else:
            paid1, paid2 = calculate_bets(round_state['action_histories'], self.uuid)
            if winners[0]['uuid'] == self.uuid:
                reward = (paid2 / 2000) * (self.hands / 500)
            else:
                reward = (paid1 / -2000) * (self.hands / 500)
            self.rewards += reward
            self.model.reward(reward)
    def done_with_reward(self, reward):
        if not self.disable_training:
            self.model.reward(reward)
            self.rewards += reward
            self.model.done()
            self.model.train_net(force = True)


def is_winner(game_result: Dict, player_name: str):
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2

def run_game(player1, name1, player2, name2, first = None, verbose = 0):
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    if first is not None:
        if first:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
    else:
        if random.random() > 0.5:
            config.register_player(name=name1, algorithm=player1)
            config.register_player(name=name2, algorithm=player2)
        else:
            config.register_player(name=name2, algorithm=player2)
            config.register_player(name=name1, algorithm=player1)
    return start_poker(config, verbose=verbose)
