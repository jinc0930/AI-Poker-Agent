from pathlib import Path
import pickle
import os
from typing import List, Literal, Optional, Self, Tuple, Union
import numpy as np
from hand import embed_hole_cards, embed_with_community
from utils import get_aggresion_freq, is_big_blind
import hnswlib

class CFRNode:
    def __init__(self, actions: List[str] = ['fold', 'raise', 'call']):
        self.actions = actions
        self.regret_sum = {a: 0.0 for a in actions}
        self.strategy_sum = {a: 0.0 for a in actions}

    def get_strategy(self, realization_weight: float) -> dict[str, float]:
        normalizing_sum = 0.0
        strategy = {}
        for a in self.actions:
            strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0.0
            normalizing_sum += strategy[a]
        if normalizing_sum > 0:
            for a in self.actions:
                strategy[a] /= normalizing_sum
        else:
            strategy = {a: 1.0/len(self.actions) for a in self.actions}
        for a in self.actions:
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self) -> dict[str, float]:
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            return {a: self.strategy_sum[a] / normalizing_sum for a in self.actions}
        else:
            return {a: 1.0/len(self.actions) for a in self.actions}

class CFRBranch:
    def __init__(self, dim: int, max_elements: int = 500_000):
        self.p: hnswlib.Index = hnswlib.Index(space = 'cosine', dim = dim)
        self.p.init_index(max_elements = max_elements)
        self.nodes: List[CFRNode] = []

    def add(self, data: np.ndarray, node: CFRNode):
        self.nodes.append(node)
        self.p.add_items(data, [len(self.nodes) -1])

    def get(self, data: np.ndarray) -> Tuple[Optional[CFRNode], Optional[float]]:
        if self.p.get_current_count() > 0:
            labels, distances = self.p.knn_query(data, k=1)
            return self.nodes[labels[0][0]], distances[0][0]
        else:
            return None, None
        
    def len(self) -> int:
        return len(self.nodes)

class CFRTree:
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance: float = tolerance
        self.branches = {
            'preflop': CFRBranch(8),
            'flop': CFRBranch(20),
            'turn': CFRBranch(20),
            'river': CFRBranch(20),
        }

    def get_or_add_node(self, street: Literal['preflop', 'flop', 'turn', 'river'], info_state: np.ndarray) -> CFRNode:
        # check if exists and return it
        if self.branches[street].len() > 0:
            node, distance = self.branches[street].get(info_state)
            if node is not None and distance < self.tolerance:
                return node

        # or create new node and return it
        new_node = CFRNode()
        self.branches[street].add(info_state, new_node)
        return new_node
    
    def size(self):
        return self.branches['preflop'].len() + self.branches['flop'].len() + self.branches['turn'].len() + self.branches['river'].len()

    def get_nearest_node(self, street: Literal['preflop', 'flop', 'turn', 'river'], info_state: np.ndarray) -> Tuple[Optional[CFRNode], Optional[float]]:
        return self.branches[street].get(info_state)

    def save(self, filename: Union[str, Path]):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Game tree saved to {filename}")

    @staticmethod
    def load(filename: Union[str, Path]) -> Self:
        if not os.path.exists(filename):
            print(f"Game tree not found: {filename}")
            return CFRTree()
        with open(filename, 'rb') as f:
            game_tree = pickle.load(f)
        print(f"Game tree loaded from {filename}")
        return game_tree

def get_info_state(round_state: dict, hole_cards: List[str], community_cards: List[str], player_uuid: str) -> np.ndarray:
    position = 1 if is_big_blind(round_state['action_histories'], player_uuid) else 0
    agg, agg_opp = get_aggresion_freq(round_state, player_uuid)
    agg_embedding = [position, agg, agg_opp]
    if round_state['street'] == 'preflop':
        hand_embedding = embed_hole_cards(hole_cards[0], hole_cards[1])
        return np.array(hand_embedding + agg_embedding, dtype=np.float32) # dim 8
    else:
        hand_embedding = embed_with_community(hole_cards, community_cards)
        return np.array(hand_embedding + agg_embedding, dtype=np.float32) # dim 20