from pathlib import Path
import re
from typing import List
from cfr import CFRNode, CFRTree, get_info_state
from players import CFRPokerPlayer, MonteCarloPlayer, RandomPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants
from utils import apply_action, extract_action, run_n_games
from tqdm import tqdm

class CFRTrainer:
    def __init__(self, emulator: Emulator, small_blind:int = 10, big_blind:int = 20, stack:int = 1000, max_raise:int = 10):
        self.emulator: Emulator = emulator
        self.small_blind: int = small_blind
        self.big_blind: int = big_blind
        self.stack: int = stack
        self.max_raise: int = max_raise
        self.game_tree: CFRTree = CFRTree()
        self.emulator.set_game_rule(player_num=2, max_round=500, small_blind_amount=small_blind, ante_amount=0)

    def is_terminal(self, game_state: dict):
        return game_state["street"] == PokerConstants.Street.FINISHED or any(p.stack <= 0 for p in game_state['table'].seats.players) or game_state['round_count'] > 500

    def get_payoff(self, game_state: dict, player_idx: int):
        player = game_state['table'].seats.players[player_idx]
        initial_stack = self.stack
        return player.stack - initial_stack

    def cfr(self, game_state: dict, events: List[dict], p0: float, p1: float):
        if self.is_terminal(game_state):
            return self.get_payoff(game_state, 0)

        current_player_idx = game_state['next_player']
        round_state = events[-1]['round_state']
        current_player = game_state['table'].seats.players[current_player_idx]
        hole_cards = sorted([str(card) for card in current_player.hole_card])
        community_cards = sorted([str(card) for card in game_state['table'].get_community_card()])

        info_state = get_info_state(round_state, hole_cards, community_cards, current_player.uuid)

        node: CFRNode = self.game_tree.get_or_add_node(round_state['street'],info_state)
        valid_actions = list(map(extract_action, events[-1]['valid_actions']))

        strategy = node.get_strategy(p0 if current_player_idx == 0 else p1)
        action_values = {}
        node_value = 0

        for action in valid_actions:
            next_game_state, next_events = apply_action(self.emulator, game_state, action)
            action_values[action] = self.cfr(next_game_state, next_events,
                                             p0 * strategy[action] if current_player_idx == 0 else p0,
                                             p1 * strategy[action] if current_player_idx == 1 else p1)
            node_value += strategy[action] * action_values[action]

        for action in valid_actions:
            if current_player_idx == 0:
                regret = action_values[action] - node_value
                node.regret_sum[action] += p1 * regret
            else:
                regret = node_value - action_values[action]
                node.regret_sum[action] += p0 * regret
        return node_value

    def train(self, iterations: int = 1000, recover_last_checkpoint: bool = True):
        checkpoints_dir = Path("./checkpoints")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        start = 0
        if recover_last_checkpoint:
            checkpoints = sorted(checkpoints_dir.glob("*"), key=lambda x: x.name, reverse=True)
            if checkpoints:
                self.game_tree = CFRTree.load(checkpoints[0])
                done = re.search(r'\d+', checkpoints[0].name).group(0)
                start += int(done)

        for i in tqdm(range(start, iterations + 1)):
            if i > 0 and (i % 1000 == 0 or i == iterations - 1):
                self.game_tree.save(f"./checkpoints/cfr_checkpoint_{i}.pickle")
                # print(f"Iteration {i}/{iterations}")
                print(f"Game tree size: {self.game_tree.size()} nodes")
            players_info = {
                "uuid-1": { "name": "P1", "stack": 1000 },
                "uuid-2": { "name": "P2", "stack": 1000 }
            }
            game_state = self.emulator.generate_initial_game_state(players_info)
            game_state, events = self.emulator.start_new_round(game_state)
            self.cfr(game_state, events, 1.0, 1.0)

        return self.game_tree

if __name__ == "__main__":
    print("Training CFR agent...")
    emulator = Emulator()
    trainer = CFRTrainer(emulator)
    game_tree = trainer.train(iterations=10_000, recover_last_checkpoint=True)
    game_tree.save('./cfr.pickle')
    print('Done!')

    print("Evaluating CFR agent...")
    cfr_agent = CFRPokerPlayer.load_from_file('./cfr.pickle')
    result = run_n_games(cfr_agent, 'CFR', RandomPlayer(0.6), 'RandomPlayer', n_games=1000)
    print(f'WR against RandomPlayer: {result}')