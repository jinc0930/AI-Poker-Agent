import os
import random
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Callable, List
import uuid
import torch
from utils import AITrainer, BluffPlayer, CallPlayer, RandomPlayer, is_winner, run_game
from model import Hyperparams
from torch.utils.tensorboard import SummaryWriter
from pypokerengine.players import BasePokerPlayer
from typing import Optional


@dataclass
class Agent:
    name: str
    display_name: Optional[str] = None
    load: Optional[Callable[[], BasePokerPlayer]] = None
    wins: int = 0
    games: int = 0
    steps: int = 0
    hands: int = 0
    net_profit: int = 0
    folds: int = 0
    raises: int = 0
    calls: int = 0
    rewards: int = 0
    rewards_from_folding: int = 0
    is_model: bool = False
    is_frozen: bool = False
    is_adversarial: bool = False
    rank: int = 0
    hyperparams: Optional[Hyperparams] = None
    hot_filepath: Optional[str] = None

    def get_bb100(self):
        if self.hands <= 0: return 0
        return (self.net_profit/20/self.hands)*100

    def move_to_hot(self, from_path: str):
        assert from_path is not None
        if self.is_model and self.hot_filepath is None:
            if not os.path.exists('hot_models'):
                os.makedirs('hot_models')
            self.hot_filepath = f"./hot_models/{self.name}.pt"
            try:
                shutil.copy(from_path, self.hot_filepath)
            except shutil.SameFileError:
                pass
        return self

    def get_player(self):
        if self.is_model:
            if self.is_frozen:
                player = AITrainer(filename=self.hot_filepath, hyperparams=self.hyperparams)
                player.disable_training = True
                return player
            return AITrainer(filename=self.hot_filepath, hyperparams=self.hyperparams)
        else:
            return self.load()

    def save(self, loaded = None):
        if self.is_model:
            if not os.path.exists('hot_models'):
                os.makedirs('hot_models')
            if self.hot_filepath is None:
                self.hot_filepath = f"./hot_models/{self.name}.pt"
            if loaded is None:
                loaded = self.get_player()
            loaded.model.save(filename = self.hot_filepath)

    def copy_to(self, filepath: str):
        if self.is_model:
            try:
                shutil.copy(self.hot_filepath, filepath)
            except shutil.SameFileError:
                pass

    def clone(self):
        display_name = f"{'Frozen' if not self.name.startswith('Frozen') else ''}{self.name.split('_')[0]}_{self.steps}"
        cloned = Agent(
                name = f"{uuid.uuid4()}",
                display_name = display_name,
                is_frozen = True,
                is_model = self.is_model,
                hyperparams = self.hyperparams,
                wins=self.wins,
                games=self.games,
                steps=self.steps,
                hands=self.hands,
                net_profit=self.net_profit,
                load=self.load
            ).move_to_hot(self.hot_filepath)
        return cloned

    def get_display_name(self):
        return self.display_name if self.display_name is not None else self.name

    def delete(self):
        if self.is_model and self.hot_filepath is not None:
            try:
                os.unlink(self.hot_filepath)
            except:
                print("Error deleting file: " + self.hot_filepath)

def choose_opponent(current_agent, agents):
    opponents = [p for p in agents if p.name != current_agent.name]
    return secrets.choice(opponents)

def choose_strong_opponent(current_agent, agents):
    weights = [0.5 ** i for i in range(len(agents))]
    return random.choices(agents, weights=weights, k=1)[0]

def choose_least_games_opponent(current_agent, agents):
    games_played = [agent.games for agent in agents]
    max_games = max(games_played)
    weights = [max_games - games + 1 for games in games_played]
    return random.choices(agents, weights=weights, k=1)[0]

class Trainer():
    def __init__(
        self,
        iterations=1_000_000,
        snapshot_steps=500,
        replicate_window=64,
        replicate_rank=2,
        stats_decay=0.95,
        log_dir='logs/arena_' + str(int(time.time())),
        league_size = 16,
        snapshot_dir = 'league',
        load_snapshot = True,
        start_replicate_after = 10_000,
        main_agents: List[Agent] = [],
        frozen_agents: List[Agent] = [],
        replicate_bb100 = 10,
    ) -> None:
        self.next_agent = 0
        self.global_steps = 0
        self.lock = threading.Lock()
        self.log_dir = log_dir
        self.snapshot_dir = snapshot_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.load_snapshot = load_snapshot
        self.iterations = iterations
        self.snapshot_steps = snapshot_steps
        self.replicate_window = replicate_window
        self.replicate_rank = replicate_rank
        self.start_replicate_after = start_replicate_after
        self.league_size = league_size
        self.main_agents = main_agents
        self.frozen_agents = frozen_agents
        self.stats_decay = stats_decay
        self.replicate_bb100 = replicate_bb100
        self.load_agents()
    def load_agents(self):
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        filenames = os.listdir(self.snapshot_dir)
        filenames.sort(reverse=True, key=str.lower)
        if self.load_snapshot and len(filenames) > 0:
            for filename in filenames:
                start, end = filename.rfind('_'), filename.rfind('.')
                steps = filename[start+1:end]
                if not steps.isdigit(): continue
                is_frozen = filename.startswith('Frozen');
                if is_frozen and len(self.frozen_agents) >= self.league_size: continue
                pool = self.frozen_agents if is_frozen else self.main_agents
                pool.append(Agent(
                    name = filename.split('.')[0],
                    is_model = True,
                    is_frozen = is_frozen,
                    hyperparams = Hyperparams(),
                    steps = int(steps),
                ).move_to_hot(f"./{self.snapshot_dir}/{filename}"))

    def print_rank(self):
        all_sorted = sorted(self.main_agents + self.frozen_agents, key=lambda x: x.get_bb100(), reverse=True)
        print("\nRANK:")
        for i, a_agent in enumerate(all_sorted):
            a_agent.rank = i
            print(f"{a_agent.get_display_name():<30} {int(a_agent.get_bb100()):<3}")

    def benchmark(self):
        agent = self.main_agents[0]
        p = agent.get_player()
        p.disable_training = True
        with torch.no_grad():
            game_result = run_game(
                p,
                agent.name,
                Agent('CallPlayer', load = lambda: CallPlayer(), is_frozen = True ),
                'CallPlayer')
        is_win, chips, opponent_chips = is_winner(game_result, agent.name)
        print('Against CallPlayer: ', chips - 1000)

    def train(self):

        print('training', self.main_agents)
        print('frozen', self.frozen_agents)

        assert self.main_agents
        assert self.frozen_agents

        while True:
            # match making
            if (self.global_steps >= self.iterations or not self.main_agents): break
            agent = self.main_agents[self.next_agent]
            self.next_agent = (self.next_agent + 1) % len(self.main_agents)
            opponent = choose_strong_opponent(agent, self.frozen_agents) if agent.steps % 2 == 0 else choose_least_games_opponent(agent, self.frozen_agents)
            if opponent is None: break
            p1, p2 = agent.get_player(), opponent.get_player()

            # play
            game_result = run_game(p1, agent.name, p2, opponent.name, first=agent.steps % 2 == 0)
            is_win, chips, opponent_chips = is_winner(game_result, agent.name)

            # post play
            agent.steps += 1
            agent.wins = agent.wins * self.stats_decay
            if is_win: agent.wins += 1
            agent.games = agent.games * self.stats_decay + 1
            agent.net_profit = agent.net_profit * self.stats_decay + (chips - 1000)
            agent.hands = agent.hands * self.stats_decay + p1.hands
            agent.folds = agent.folds * self.stats_decay + p1.folds
            agent.raises = agent.raises * self.stats_decay + p1.raises
            agent.calls = agent.calls * self.stats_decay + p1.calls
            if agent.is_model:
                # reward the ratio of folds to encourage folds even against weak/tight opponents
                p1.done_with_reward(1 if is_win else -1)
                #p1.done_with_reward(agent.folds/agent.hands if is_win else -1)
                #p1.done_with_reward((chips - 1000)/1000)
            agent.rewards = agent.rewards * self.stats_decay + p1.rewards
            agent.rewards_from_folding = agent.rewards_from_folding * self.stats_decay + p1.rewards_from_folding


            if opponent.is_frozen:
                opponent.hands = opponent.hands * self.stats_decay + p2.hands
                opponent.games = opponent.games * self.stats_decay + 1
                opponent.net_profit = opponent.net_profit * self.stats_decay + (opponent_chips - 1000)
                if not is_win: opponent.wins += 1

            if agent.is_model and not agent.is_frozen:
                # metrics
                self.writer.add_scalar(agent.name + '/BB/100', agent.get_bb100(), agent.steps)
                self.writer.add_scalar(agent.name + '/Chips', chips, agent.steps)
                self.writer.add_scalar(agent.name + '/WinRate', agent.wins/(agent.games + 1), agent.steps)
                self.writer.add_scalar(agent.name + '/FoldRate', agent.folds/(agent.hands + 1), agent.steps)
                self.writer.add_scalar(agent.name + '/CallRate', agent.calls/(agent.hands + 1), agent.steps)
                self.writer.add_scalar(agent.name + '/RaiseRate', agent.raises/(agent.hands + 1), agent.steps)
                self.writer.add_scalar(agent.name + '/RewardRate', agent.rewards/(agent.games + 1), agent.steps)
                # self.writer.add_scalar(agent.name + '/FoldRewardRate', agent.rewards_from_folding/(agent.games + 1), agent.steps)

            # Save after training
            agent.save(p1)

            # Replicate new agents when top N = replicate_rank
            if agent.is_model and agent.steps > self.start_replicate_after and \
             (agent.steps % self.replicate_window == 0 and\
              agent.rank <= self.replicate_rank and\
              agent.get_bb100() > self.replicate_bb100):
                cloned_agent = agent.clone()
                cloned_agent.games = 0
                cloned_agent.wins = 0

                self.frozen_agents.append(cloned_agent)
                # print(f'replicated {agent.name} to {cloned_agent.display_name()}')

            # Global updates
            if self.global_steps % 1000 == 0:
                self.print_rank()
                #self.benchmark()

            if self.global_steps > 0 and self.global_steps % self.snapshot_steps == 0:
                print(f"\nIteration {self.global_steps}/{self.iterations}")

                # Update/trim league
                self.frozen_agents = sorted(self.frozen_agents, key=lambda x: x.get_bb100(), reverse=True)
                cut = max(0, self.league_size - len(self.main_agents))
                to_be_removed = self.frozen_agents[cut:]
                keep_agents = self.frozen_agents[:cut]
                self.frozen_agents = sorted(keep_agents, key=lambda x: x.get_bb100(), reverse=True)
                for a_agent in to_be_removed:
                    if a_agent.games > 3 or a_agent.get_bb100() < 0:
                        a_agent.delete()
                    else:
                        keep_agents.append(a_agent)
                self.frozen_agents = keep_agents

                # Snapshot the league
                if not os.path.exists(self.snapshot_dir):
                    os.makedirs(self.snapshot_dir)
                else:
                    for filename in os.listdir(self.snapshot_dir):
                        file_path = os.path.join(self.snapshot_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path) # Delete file or link
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path) # Delete subdirectory
                        except Exception as e:
                            print(f"failed to delete {file_path}: {e}")
                for a_agent in self.main_agents:
                    if a_agent.is_model:
                        a_agent.copy_to(f"./{self.snapshot_dir}/{a_agent.get_display_name().split('_')[0]}_{a_agent.steps}.pt")
                for a_agent in self.frozen_agents:
                    if a_agent.is_model:
                        a_agent.copy_to(f"./{self.snapshot_dir}/{a_agent.get_display_name().split('_')[0]}_{a_agent.steps}.pt")

                print(f'league snapshot taken at ./{self.snapshot_dir}\n')
            self.global_steps += 1

        self.writer.close()
        return self.main_agents

if __name__ == '__main__':
    # Main agents
    main_agents = [
        Agent(
            name = 'Star5',
            is_model = True,
            hyperparams = Hyperparams(),
        ),
    ]

    # Frozen agents
    frozen_agents = [
        Agent('CallPlayer', load = lambda: CallPlayer(), is_frozen = True ),
        Agent('BluffPlayer', load = lambda: BluffPlayer(), is_frozen = True ),
    ]

    for fold, raisee in [(0.001, 0)]:
        a = RandomPlayer()
        a.set_action_ratio(fold, raisee, 1 - (fold + raisee))
        frozen_agents.append(Agent(f'RandomPlayer{fold}-{raisee}', load = lambda: RandomPlayer(), is_frozen = True ))

    # If load_snapshot is True you might not want to use frozen and main_agents
    Trainer(
        iterations=1_000_000,
        snapshot_steps=1000,
        replicate_window=1024,
        replicate_rank=0, # top 1
        log_dir='logs/arena_' + str(int(time.time())),
        league_size = 64,
        snapshot_dir = 'star5',
        load_snapshot = True,
        start_replicate_after = 1000,
        main_agents=main_agents,
        frozen_agents=frozen_agents
    ).train()