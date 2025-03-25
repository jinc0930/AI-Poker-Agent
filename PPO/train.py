from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
import random as rand
from model import PPO, Hyperparams
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from encoder import encode
import os
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from torch.utils.tensorboard import SummaryWriter
import random as rand
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict
import threading
import secrets
import os
import shutil
import re

def get_stacks(seats, your_uuid):
    your_stack = next(s['stack'] for s in seats if s['uuid'] == your_uuid)
    opponent_stack = next(s['stack'] for s in seats if s['uuid'] != your_uuid)
    return your_stack, opponent_stack

def is_big_blind(action_histories, player_uuid):
    for action in action_histories['preflop']:
        if action["action"] == "BIGBLIND" and action["uuid"] == player_uuid:
            return True
    return False

class CallPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [FOLD, CALL, RAISE]
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hands = 0

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [FOLD, CALL, RAISE]
        r = rand.random()
        if r < 0.33:
            action, amount = valid_actions[0]["action"], valid_actions[0]["amount"]
        elif r < 0.66:
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:
            action, amount = valid_actions[2]["action"], valid_actions[2]["amount"]["min"]
        return action, amount

    def receive_game_start_message(self, game_info):
        self.hands = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class BluffingPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        r = random.random()

        # Bluffing logic: Raise more often regardless of hand strength
        if r < 0.6:  # 60% chance to raise
            action, amount = valid_actions[2]["action"], valid_actions[2]["amount"]["min"]
        elif r < 0.8:  # 20% chance to call
            action, amount = valid_actions[1]["action"], valid_actions[1]["amount"]
        else:  # 20% chance to fold
            action, amount = valid_actions[0]["action"], valid_actions[0]["amount"]

        return action, amount

    def receive_game_start_message(self, game_info):
        self.hands = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1

class BalancedPlayer(BasePokerPlayer):
    def hand_strength(self, hole_card):
        # Map card ranks to values
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        card1, card2 = hole_card[0], hole_card[1]
        rank1, suit1 = card1[1], card1[0]
        rank2, suit2 = card2[1], card2[0]

        # Base score: sum of the card values
        score = rank_values[rank1] + rank_values[rank2]

        # Bonus for pairs
        if rank1 == rank2:
            score += 10

        # Bonus for suited cards
        if suit1 == suit2:
            score += 4

        # Bonus for connected cards (difference of 1 or same rank)
        if abs(rank_values[rank1] - rank_values[rank2]) == 1:
            score += 3

        return score

    def declare_action(self, valid_actions, hole_card, round_state):
        score = self.hand_strength(hole_card)
        if score >= 24:  # strong hand threshold
            # Try to raise with the min bet if available
            for action_info in valid_actions:
                if action_info["action"] == "raise":
                    return action_info["action"], action_info["amount"]["min"]
            # If no raise available, call.
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"], action_info["amount"]
        elif score >= 18:  # medium hand threshold
            # Prefer to call, but sometimes raise to mix things up
            if random.random() < 0.3:
                for action_info in valid_actions:
                    if action_info["action"] == "raise":
                        return action_info["action"], action_info["amount"]["min"]
            for action_info in valid_actions:
                if action_info["action"] == "call":
                    return action_info["action"], action_info["amount"]
        else:
            # Weak hand: mostly fold, but occasionally call if the cost is zero.
            for action_info in valid_actions:
                if action_info["action"] == "call" and action_info["amount"] == 0:
                    return action_info["action"], action_info["amount"]
            return valid_actions[0]["action"], valid_actions[0]["amount"]

    def receive_game_start_message(self, game_info):
        self.hands = 0

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        self.hands += 1


class AITrainer(BasePokerPlayer):
    def __init__(self, filename=None, hyperparams=Hyperparams(), disable_training=False):
        super().__init__()
        self.model = PPO(filename=filename, hyperparams=hyperparams)
        self.state = None
        self.disable_training = disable_training
        self.hands = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        if not self.disable_training: self.model.train_net()
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

        if self.state is None:
            self.state = state_prime

        prob = self.model.pi(torch.tensor(self.state))
        action = Categorical(prob).sample().item()
        if not self.disable_training:
            reward = 0
            self.model.put_data((self.state, action, reward, state_prime, prob[action].item(), False))
        self.state = state_prime

        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        final = valid_actions[action]
        if action == 2:
            return final["action"], final["amount"]["min"]
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
        self.hands += 1
        if not self.disable_training:
            if winners[0]['uuid'] == self.uuid:
                self.model.reward(round_state['pot']['main']['amount'] / 2000)
            else:
                self.model.reward(round_state['pot']['main']['amount'] / -2000)
    def force_train(self):
        self.model.train_net(force = True)

def run_game(player1, name1, player2, name2, first = True, verbose = 0):
    config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)
    if first:
        config.register_player(name=name1, algorithm=player1)
        config.register_player(name=name2, algorithm=player2)
    else:
        config.register_player(name=name2, algorithm=player2)
        config.register_player(name=name1, algorithm=player1)
    return start_poker(config, verbose=verbose)

def is_winner(game_result: Dict, player_name: str):
    players = game_result['players']
    winner = max(players, key=lambda player: player['stack'])
    stack = next(player['stack'] for player in players if player['name'] == player_name)
    stack2 = next(player['stack'] for player in players if player['name'] != player_name)
    won = winner['name'] == player_name
    return won, stack, stack2

@dataclass
class Agent:
    name: str
    load: None | Callable[[], BasePokerPlayer] = None
    wins: int = 0
    games: int = 0
    steps: int = 0
    wins_vs_frozen: int = 0
    games_vs_frozen: int = 0
    hands: int = 0
    chips: int = 0
    is_model: bool = False
    is_frozen: bool = False
    hyperparams: Hyperparams | None = None
    filename: str | None = None
    hot_filename: str | None = None
    lock = threading.Lock()

    def get_bb100(self):
        return (self.chips/20/(self.hands + 1))*100

    def __post_init__(self):
        if self.is_model:
            if self.hot_filename is None:
                self.hot_filename = f"./models/{self.name}.pt"
                self.save()
            else:
                self.save(AITrainer(filename=self.filename, hyperparams=self.hyperparams))
                self.hot_filename = f"./models/{self.name}.pt"

    def get_player(self):
        if self.is_model:
            with self.lock:
                if self.is_frozen:
                    player = AITrainer(filename=self.hot_filename, hyperparams=self.hyperparams)
                    player.disable_training = True
                    return player
                return AITrainer(filename=self.hot_filename, hyperparams=self.hyperparams)
        else:
            return self.load()

    def save(self, loaded = None):
        if self.is_model:
            if not os.path.exists('models'):
                os.makedirs('models')
            if loaded is None:
                loaded = self.get_player()
            with self.lock:
                loaded.model.save(filename = self.hot_filename)

    def clone(self, id = ''):
        name = f"{'Frozen' if not self.name.startswith('Frozen') else ''}{self.name.split('_')[0]}_{self.steps}"
        if id != '':
            name = re.sub(r'\d+', id, name, 1)
        cloned = Agent(
                name = name,
                is_frozen = True,
                is_model = self.is_model,
                hyperparams = self.hyperparams,
                wins=self.wins,
                games=self.games,
                steps=self.steps,
                hands=self.hands,
                chips=self.chips,
                wins_vs_frozen=self.wins_vs_frozen,
                games_vs_frozen=self.games_vs_frozen,
            )
        return cloned
    def delete(self):
        if self.is_model:
            with self.lock:
                os.unlink(self.hot_filename)

def choose_opponent(current_agent, agents):
    opponents = [p for p in agents if p.name != current_agent.name]
    return secrets.choice(opponents)

def train(
    iterations=1_000_000,
    snapshot_steps=800,
    replicate_steps=1000,
    log_dir='results/arena_' + str(int(time.time())),
    league_size = 32,
    use_league = True,
    snapshot_dir = 'league',
    n_agents = 3
):
    # Initialize everything
    lock = threading.Lock()
    writer = SummaryWriter(log_dir=log_dir)

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    global global_iterations
    global next_agent
    global main_agents
    global frozen_agents

    # Inititalize agents
    main_agents = []
    frozen_agents = []

    # static_agents = [
    #     Agent('CallPlayer', load = lambda: CallPlayer(), is_frozen = True ),
    #     Agent('RandomPlayer', load = lambda: RandomPlayer(), is_frozen = True ),
    #     Agent('BluffingPlayer', load = lambda: BluffingPlayer(), is_frozen = True ),
    #     Agent('BalancedPlayer', load = lambda: BalancedPlayer(), is_frozen = True ),
    # ]

    league = os.listdir(snapshot_dir)
    if use_league and len(league) > 0:
        for filename in league:
            start, end = filename.rfind('_'), filename.rfind('.')
            steps = filename[start+1:end]
            if not steps.isdigit(): continue
            is_frozen = filename.startswith('Frozen');
            pool = frozen_agents if is_frozen else main_agents
            pool.append(Agent(
                name = filename.split('.')[0],
                is_model = True,
                is_frozen = is_frozen,
                filename = os.path.join(snapshot_dir, filename),
                hyperparams = Hyperparams(),
                steps = int(steps),
            ))
    else:
        for i in range(n_agents):
            main_agents.append(Agent(
                name = f'MainAgent{i}',
                is_model = True,
                hyperparams = Hyperparams(),
            ))

    assert league_size > len(main_agents), "League size must be greater than number of agents"
    assert n_agents > 0, "Number of agents must be greater than 0"
    assert n_agents == len(main_agents), "Number of agents must be equal to number of expected agents"

    # Increase league size if needed
    current_len = len(frozen_agents) + len(main_agents)
    if current_len < league_size:
        new_frozen_agents = []
        for i in range(league_size):
            new_frozen_agents.append(Agent('BalancedPlayer' + str(i), load = lambda: BalancedPlayer(), is_frozen = True ))
            # if frozen_agents:
            #     new_frozen_agents.append(frozen_agents[i % len(frozen_agents)].clone('c' + str(i)))
            # else:
            #     new_frozen_agents.append(main_agents[i % len(main_agents)].clone('c' + str(i)))
        frozen_agents.extend(new_frozen_agents)

    next_agent = 0
    global_iterations = 0

    def train_agent(is_main = False):
        global global_iterations
        global next_agent
        global main_agents
        global frozen_agents
        i = 0
        while True:
            # match making
            with lock:
                if (global_iterations >= iterations or not main_agents): break
                agent = main_agents[next_agent]
                next_agent = (next_agent + 1) % len(main_agents)
                opponent = choose_opponent(agent, main_agents + frozen_agents)
                if opponent is None: break
                p1, p2 = agent.get_player(), opponent.get_player()

            # play
            game_result = run_game(p1, agent.name, p2, opponent.name, first=agent.steps % 2 == 0)
            is_win, chips, opponent_chips = is_winner(game_result, agent.name)

            # post play
            with lock:
                # update
                if is_win: agent.wins += 1
                agent.games += 1
                agent.steps += 1
                agent.chips += (chips - opponent_chips)
                agent.hands += p1.hands
                p1.force_train()

                if opponent.is_frozen:
                    opponent.hands += p2.hands
                    opponent.games += 1
                    opponent.chips += (opponent_chips - chips)
                    if not is_win: opponent.wins += 1
                    if is_win: agent.wins_vs_frozen += 1
                    agent.games_vs_frozen += 1

                # metrics
                if agent.is_model:
                    writer.add_scalar(agent.name + '/BB/100', agent.get_bb100(), agent.steps)
                writer.add_scalar(agent.name + '/Chips', chips, agent.steps)
                writer.add_scalar(agent.name + '/WR', agent.wins/(agent.games + 1), agent.steps)
                writer.add_scalar(agent.name + '/WR_vs_Frozen', agent.wins_vs_frozen/(agent.games_vs_frozen + 1), agent.games_vs_frozen)

                # Save after training
                agent.save(p1)

                # Replicate new agents
                if agent.is_model and agent.steps > 0 and agent.steps % replicate_steps == 0:
                    frozen_agents.append(agent.clone())
                    #print(f'replicated {agent.name}')

                # Global updates
                if is_main and i > 0 and i % snapshot_steps == 0:
                    print(f"\nIteration {global_iterations}/{iterations}")

                    # Update/trim league
                    frozen_agents = sorted(frozen_agents, key=lambda x: x.get_bb100(), reverse=True)
                    cut = max(0, league_size - len(main_agents))
                    removed_agents = frozen_agents[cut:]
                    for a_agent in removed_agents:
                        a_agent.delete()
                    keep_agents = frozen_agents[:cut]
                    frozen_agents = keep_agents
                    all_sorted = sorted(main_agents + frozen_agents, key=lambda x: x.get_bb100(), reverse=True)

                    # Report all agents
                    for a_agent in all_sorted:
                        print(f"{a_agent.name:<30} {int(a_agent.get_bb100()):<3}")

                    # Snapshot the league
                    if not os.path.exists(snapshot_dir):
                        os.makedirs(snapshot_dir)
                    else:
                        for filename in os.listdir(snapshot_dir):
                            file_path = os.path.join(snapshot_dir, filename)
                            try:
                                if os.path.isfile(file_path) or os.path.islink(file_path):
                                    os.unlink(file_path) # Delete file or link
                                elif os.path.isdir(file_path):
                                    shutil.rmtree(file_path) # Delete subdirectory
                            except Exception as e:
                                print(f"failed to delete {file_path}: {e}")
                    for a_agent in main_agents:
                        if a_agent.is_model:
                            p1.model.save(f"./{snapshot_dir}/{a_agent.name.split('_')[0]}_{a_agent.steps}.pt")
                    for a_agent in frozen_agents:
                        if a_agent.is_model:
                            p1.model.save(f"./{snapshot_dir}/{a_agent.name.split('_')[0]}_{a_agent.steps}.pt")

                    # Reset the league.
                    for a_agent in all_sorted:
                        factor = max(agent.games, 1)
                        a_agent.wins //= factor
                        a_agent.games //= factor
                        a_agent.hands //= factor
                        a_agent.chips //= factor
                    print(f'league snapshot taken at ./{snapshot_dir}\n')
                global_iterations += 1
            i += 1

    # Train
    threads = []
    n_threads = min(os.cpu_count(), n_agents)
    print(f"spawning {n_threads} threads")

    for i in range(n_threads):
        is_main = i == 0
        thread = threading.Thread(target = lambda is_main=is_main: train_agent(is_main))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    writer.close()
    return main_agents

if __name__ == '__main__':
    train(
        iterations=1_000_000,
        snapshot_steps=500,
        replicate_steps=32,
        league_size=32
    )