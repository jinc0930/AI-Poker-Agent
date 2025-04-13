import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
import copy
import os
import shutil
import re
import uuid
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Hyperparams
from utils import AITrainer, BluffPlayer, CallPlayer, MonteCarloPlayer, is_winner, linear_schedule, run_game
from pypokerengine.players import BasePokerPlayer

@dataclass
class Agent:
    name: str
    display_name: Optional[str] = None
    load: Optional[Callable[[], BasePokerPlayer]] = None
    wins: int = 0
    games: int = 0
    episodes: int = 0
    hands: int = 0
    net_profit: int = 0
    folds: int = 0
    raises: int = 0
    calls: int = 0
    rewards: int = 0
    is_model: bool = False
    is_frozen: bool = False
    rank: int = 0
    wr: float = 0
    played_at: int = 0
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
                player.is_opponent = True
                return player
            hyperparams = self.hyperparams if self.hyperparams is not None else Hyperparams()
            # entropy schedule
            hyperparams.entropy_coeff = linear_schedule(hyperparams.entropy_coeff, 0.001, self.episodes, 500_000)
            return AITrainer(filename=self.hot_filepath, hyperparams=hyperparams)
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
        display_name = f"{'Frozen' if not self.name.startswith('Frozen') else ''}{self.name.split('_')[0]}_{self.episodes}"
        cloned = Agent(
                name = f"{uuid.uuid4()}",
                display_name = display_name,
                is_frozen = True,
                is_model = self.is_model,
                hyperparams = self.hyperparams,
                wins=self.wins,
                games=self.games,
                episodes=self.episodes,
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

class PFSP():
    def __init__(
        self,
        writer=None,
        population_size = 100,
        snapshot_dir = 'population',
        load_snapshot = True,
        main_agents: List[Agent] = [],
        frozen_agents: List[Agent] = [],
        evaluate_agents: List[Agent] = [],
        prioritization_alpha = 3,
    ) -> None:
        self.next_agent = 0
        self.snapshot_dir = snapshot_dir
        self.writer = writer
        self.load_snapshot = load_snapshot
        self.population_size = population_size
        self.main_agents = main_agents
        self.frozen_agents = frozen_agents
        self.prioritization_alpha = prioritization_alpha
        self.load_agents()

        if not evaluate_agents:
            self.evaluate_agents = [
                # Agent('RandomPlayer', load = lambda: RandomPlayer(), is_frozen = True ),
                Agent('CallPlayer', load = lambda: CallPlayer(), is_frozen = True ),
                Agent('BluffPlayer', load = lambda: BluffPlayer(), is_frozen = True ),
                Agent('MonteCarloPlayer-0', load = lambda: MonteCarloPlayer(0), is_frozen = True ),
                Agent('MonteCarloPlayer-1', load = lambda: MonteCarloPlayer(0.5), is_frozen = True ),
                Agent('MonteCarloPlayer-0.5', load = lambda: MonteCarloPlayer(1), is_frozen = True ),
            ]

        self.exploiter_agents = [
            # Agent(
            #     name = 'Exploiter',
            #     is_model = True,
            #     hyperparams = Hyperparams(
            #         learning_rate=0.001,
            #         eps_clip=0.2,
            #     ),
            # ),
        ]
        #self.exploiter_agents[0].save()
    def load_agents(self):
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        filenames = os.listdir(self.snapshot_dir)
        filenames.sort(reverse=True, key=str.lower)
        if self.load_snapshot and len(filenames) > 0:
            for filename in filenames:
                start, end = filename.rfind('_'), filename.rfind('.')
                episodes = filename[start+1:end]
                if not episodes.isdigit(): continue
                is_frozen = filename.startswith('Frozen');
                if is_frozen and len(self.frozen_agents) >= self.population_size: continue
                pool = self.frozen_agents
                if not is_frozen:
                    pool = self.main_agents
                elif filename.startswith('Exploiter'):
                    pool = self.exploiter_agents
                pool.append(Agent(
                    name = filename.split('.')[0],
                    is_model = True,
                    is_frozen = is_frozen,
                    hyperparams = Hyperparams(),
                    episodes = int(episodes),
                ).move_to_hot(f"./{self.snapshot_dir}/{filename}"))

    def print_rank(self):
        all_sorted = sorted(self.frozen_agents, key=lambda x: x.wr)
        print("\nAgent VS Opponents (WR):")
        for i, a_agent in enumerate(all_sorted):
            a_agent.rank = i
            print(f"{a_agent.get_display_name():<30} {(a_agent.wr * 100.):<6.2f}% {a_agent.games:<5}")


    def add_frozen_agent(self, agent: Agent):
        self.frozen_agents.append(agent)

    def evaluate(self, episodes = 100):
        agent = self.main_agents[0]
        wins = 0
        chips = 0

        for frozen in self.evaluate_agents:
            for _ in range(episodes):

                p = agent.get_player()
                p.disable_training = True

                with torch.no_grad():
                    game_result = run_game(p, agent.name, frozen.get_player(), frozen.name, first=agent.episodes % 2 == 0)
                    is_win, chips, _ = is_winner(game_result, agent.name)
                    wins += 1 if is_win else 0
                    chips += (chips - 1000)

        div = episodes * len(self.frozen_agents)
        return wins / div, chips / div


    def select_opponent(self, win_rate_weight=0.9, recency_weight=0.1, temperature=0.1):
        if len(self.frozen_agents) == 1:
            return self.frozen_agents[0]

        # Ensure weights sum to 1
        total_weight = win_rate_weight + recency_weight
        win_rate_weight = win_rate_weight / total_weight
        recency_weight = recency_weight / total_weight

        agent = self.main_agents[0]
        scores = np.zeros(len(self.frozen_agents))

        for idx, opp in enumerate(self.frozen_agents):
            # Win rate component (prioritize opponents that the player loses to)
            win_rate_score = 1.0 - opp.wr

            # Recency component (prioritize opponents not played recently)
            iterations_since_played = agent.episodes - opp.played_at
            max_expected_iterations = 3_000
            recency_score = min(iterations_since_played / max_expected_iterations, 1.0)

            # Combine scores with configurable weights
            scores[idx] = win_rate_weight * win_rate_score + recency_weight * recency_score

        # Apply softmax with temperature to get probabilities
        scores = scores - np.max(scores)  # For numerical stability
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Sample an opponent based on the probabilities
        opponent_idx = np.random.choice(len(self.frozen_agents), p=probs)
        return self.frozen_agents[opponent_idx]

    # Exponential moving average
    def ema(self, old_value, new_value, alpha = 0.01):
        return (1 - alpha) * old_value + alpha * new_value

    def update_stats(self, agent: Agent, p1: AITrainer, is_win: bool, chips: int):
        agent.episodes += 1
        agent.games += 1
        agent.wins += 1 if is_win else 0
        agent.net_profit += chips - 1000
        agent.hands = self.ema(agent.hands, agent.hands + p1.hands)
        agent.folds = self.ema(agent.folds, p1.folds / p1.hands)
        agent.raises = self.ema(agent.raises, p1.raises  / p1.hands)
        agent.calls = self.ema(agent.calls, p1.calls  / p1.hands)
        if agent.is_model:
            #reward = 1 if is_win else -1
            #reward = agent.folds/agent.hands if is_win else -1
            #reward = ((chips - 1000)/1000)
            reward = 1 if is_win else -1
            p1.done_with_reward(reward)
            agent.rewards = self.ema(agent.rewards, p1.rewards)
            # Save after training
            agent.save(p1)
            if not agent.is_frozen:
                # metrics
                self.writer.add_scalar(agent.name + '/FoldRate', agent.folds, agent.episodes)
                self.writer.add_scalar(agent.name + '/CallRate', agent.calls, agent.episodes)
                self.writer.add_scalar(agent.name + '/RaiseRate', agent.raises, agent.episodes)
                self.writer.add_scalar(agent.name + '/RewardsRate', agent.rewards, agent.episodes)

    def train_iteration(self, episodes=100):
        agent = self.main_agents[0]
        opponent = self.select_opponent()

        wins = 0
        for _ in range(episodes):
            p1, p2 = agent.get_player(), opponent.get_player()
            result = run_game(p1, agent.name, p2, opponent.name)
            is_win, chips, opponent_chips = is_winner(result, agent.name)
            wins += 1 if is_win else 0

            # post play
            self.update_stats(agent, p1, is_win, chips)
            if opponent.is_frozen:
                opponent.games += 1
                if not is_win: opponent.wins += 1
                opponent.hands = self.ema(opponent.hands, opponent.hands + p2.hands)
                opponent.net_profit = self.ema(opponent.net_profit, opponent_chips - 1000)
                opponent.played_at = agent.episodes

        # update agent_wr against opponent
        new_wr = wins / episodes
        opponent.wr = self.ema(opponent.wr, new_wr, 0.25)

        # train exploiter
        if agent.episodes % 3 == 0 and self.exploiter_agents:
            exploiter = self.exploiter_agents[0]
            for _ in range(episodes):
                p1, p2 = exploiter.get_player(), agent.get_player()
                p2.disable_training = True
                result = run_game(p1, exploiter.name, p2, agent.name)
                is_win, chips, agent_chips = is_winner(result, exploiter.name)
                self.update_stats(exploiter, p1, is_win, chips)

        return new_wr

    # replicate main agent and add to the league
    def replicate(self, agent_type: str = "main"):
        if agent_type == "exploiter":
            if self.exploiter_agents:
                agent = self.exploiter_agents[0]
        else:
            agent = self.main_agents[0]
        cloned_agent = agent.clone()
        cloned_agent.games = 0
        cloned_agent.wins = 0
        cloned_agent.wr = 0.5
        cloned_agent.net_profit = 0
        cloned_agent.hands = 0
        cloned_agent.folds = 0
        cloned_agent.raises = 0
        cloned_agent.calls = 0
        self.frozen_agents.append(cloned_agent)

    def update_population(self):
        # Update/trim population
        self.frozen_agents = sorted(self.frozen_agents, key=lambda x: x.wr)
        cut = max(0, self.population_size - len(self.main_agents))
        to_be_removed = self.frozen_agents[cut:]
        keep_agents = self.frozen_agents[:cut]
        self.frozen_agents = sorted(keep_agents, key=lambda x: x.wr)
        for a_agent in to_be_removed:
            if a_agent.games > 10 or a_agent.wr < 0:
                a_agent.delete()
            else:
                keep_agents.append(a_agent)
        self.frozen_agents = keep_agents

        # Snapshot the population
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
                a_agent.copy_to(f"./{self.snapshot_dir}/{a_agent.get_display_name().split('_')[0]}_{a_agent.episodes}.pt")
        for a_agent in self.frozen_agents:
            if a_agent.is_model:
                a_agent.copy_to(f"./{self.snapshot_dir}/{a_agent.get_display_name().split('_')[0]}_{a_agent.episodes}.pt")

        print(f'population snapshot taken at ./{self.snapshot_dir}\n')

if __name__ == '__main__':
    # Main agents
    main_agents = [
        Agent(
            name = 'Star',
            is_model = True,
            hyperparams = Hyperparams(),
        ),
    ]

    # Frozen agents
    frozen_agents = [
        Agent('MonteCarloPlayer-0.4', load = lambda: MonteCarloPlayer(0.4), is_frozen = True ),
        Agent('MonteCarloPlayer-0.5', load = lambda: MonteCarloPlayer(0.5), is_frozen = True ),
        Agent('MonteCarloPlayer-0.6', load = lambda: MonteCarloPlayer(0.6), is_frozen = True ),
    ]

    # for fold, raisee in [(0.01, 0.5)]:
    #     a = RandomPlayer()
    #     a.set_action_ratio(fold, raisee, 1 - (fold + raisee))
    #     frozen_agents.append(Agent(f'RandomPlayer{fold}-{raisee}', load = lambda: RandomPlayer(), is_frozen = True ))


    writer = SummaryWriter(log_dir='logs/arena_' + str(int(time.time())))

    # If load_snapshot is True you might not want to use frozen and main_agents
    trainer = PFSP(
        writer=writer,
        population_size = 30,
        snapshot_dir = 'starstar',
        load_snapshot = True,
        main_agents=main_agents,
        frozen_agents=frozen_agents
    )

    # Training loop
    eval_episodes = 0
    for i in range(2000):
        wr = trainer.train_iteration(episodes=100)
        # update league
        if i > 0 and i % 10 == 0:
            trainer.print_rank()
            trainer.update_population()
            #trainer.replicate('exploiter')
            trainer.replicate('main')

        if i > 0 and i % 20 == 0:
            wr, chips = trainer.evaluate(episodes=100)
            print(f"Iteration {i}, Win rate evaluation: {(wr * 100.):.2f}%, Chips won: {chips}")
            writer.add_scalar(trainer.main_agents[0].name + '/EvalWR', wr, eval_episodes)
            writer.add_scalar(trainer.main_agents[0].name + '/EvalChips', chips, eval_episodes)
            eval_episodes += 1

        if i > 0 and i % 10 == 0:
            if i < 1000:
                difficulty = linear_schedule(0.5, 1, i, 1000)
                trainer.add_frozen_agent(
                    Agent(f'MonteCarloPlayer-{difficulty:.2f}', load = lambda: MonteCarloPlayer(difficulty), is_frozen = True )
                )
                print(f"Added opponent at iteration {i} with difficulty {difficulty:.4f}")

    writer.close()