from collections import deque
from copy import deepcopy
import math
import os
import pickle
import random
from typing import Any, Final, Generator, List, Optional, Self,Tuple
import numpy as np
from rbloom import Bloom
import tqdm
from cfr_model import DeepCFRModel
from hand import RANKS, SUITS, generate_cards, street_to_num
from players import CFRPokerPlayer, CallPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine.poker_constants import PokerConstants
from pypokerengine.engine.round_manager import RoundManager
from utils import extract_action, get_aggresion_per_round, get_initial_state, get_initial_state_with_cards, get_stacks, get_total_bettings, is_big_blind, is_facing_raise, make_hashable, run_n_games
import torch
import torch.optim as optim
import torch.nn.functional as F
import concurrent.futures

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

TRAIN_ITERATIONS = 7_000
TRAIN_EPOCHS: Final[int] = 10_000
TRAINING_PATIENCE: Final[int] = 100
TRAIN_BATCH_SIZE: Final[int] = 10_000
GAME_TRAVERSALS_PER_HAND: Final[int] = 200
FEATURE_SIZE: Final[int] = 28
LR: Final[int] = 0.001

ACTIONS: Final[List[str]] = ["fold", "raise", "call"]
ACTION_SPACE: Final[int] = len(ACTIONS)
NUM_PLAYERS: Final[int] = 2

PLAYER_ID_TO_POS = ["BB", "SB"]

def card_to_int(card_str: str) -> int:
    """Returns unique int for the card"""
    # Handle two-character and three-character card strings
    if len(card_str) == 2:
        suit = SUITS.index(card_str[0])
        rank = RANKS.index(card_str[1])
    elif len(card_str) == 3:
        suit = SUITS.index(card_str[0])
        rank = int(card_str[1:])
    else:
        raise ValueError(f"Invalid card string: {card_str}")

    return rank * 4 + suit

def convert_hand(card_strings: List[str], max_len: int = 2) -> np.ndarray:
    hand = [card_to_int(card) for card in card_strings]
    padded_hand = hand + [-1] * (max_len - len(hand))  # Pad with -1
    return np.array(padded_hand[:max_len], dtype=np.int32)

def _apply_action(emulator: Emulator, game_state: Any, action: str) -> Tuple[dict, dict]:
    updated_state, messages = RoundManager.apply_action(game_state, action)
    events = [emulator.create_event(message[1]["message"]) for message in messages]
    events = [e for e in events if e]
    if emulator._is_last_round(updated_state, emulator.game_rule):
        events += emulator._generate_game_result_event(updated_state)
    return updated_state, events

def _is_terminal(game_state: dict) -> bool:
    return game_state["street"] == PokerConstants.Street.FINISHED or any(p.stack <= 0 for p in game_state['table'].seats.players) or game_state['round_count'] > 500

def _get_payoff(round_state: dict, player_idx: int) -> float:
    is_winner = round_state['seats'][0 if player_idx == 1 else 0]['state'] == 'folded'
    paid, opp_paid = get_total_bettings(round_state['action_histories'], round_state['seats'][player_idx]['uuid'])
    return (opp_paid if is_winner else -paid) / 1000.

# Stack to pot ratio
def normalize_spr(spr: float) -> float:
    return math.log(1 + spr) / math.log(1 + 66.3333)

def get_info_set(round_state: dict, hole_cards: List[str], community_cards: List[str], player_uuid: str) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    position = 1 if is_big_blind(round_state['action_histories'], player_uuid) else 0
    agg, agg_opp = get_aggresion_per_round(round_state, player_uuid)
    street_idx = street_to_num(round_state['street'])
    stack, _ = get_stacks(round_state['seats'], player_uuid)
    spr = normalize_spr(stack / round_state['pot']['main']['amount'])
    facing_raise = 1 if is_facing_raise(round_state, player_uuid) else 0
    bettings = np.array([
        position,
        spr,
        facing_raise,
        agg / 4,
        agg_opp / 4,
    ], dtype=np.float32)

    return (
        convert_hand(hole_cards, 2),
        convert_hand(community_cards, 5),
        np.int32(street_idx),
        bettings
    )

def _get_infoset(current_player_idx: int, game_state: dict, round_state: dict) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray[float]]:
    current_player = game_state['table'].seats.players[current_player_idx]
    hole_cards = sorted([str(card) for card in current_player.hole_card])
    community_cards = sorted([str(card) for card in game_state['table'].get_community_card()])
    infoset = get_info_set(round_state, hole_cards, community_cards, current_player.uuid)
    return infoset

def fake_info_set(position: int = 0) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    agg, agg_opp = random.randint(0, 4), random.randint(0, 4)
    street_idx = random.randint(0, 3)
    pot = random.randint(30, 2000)
    stack = random.randint(10, max(1990 - pot - 10, 10))
    spr = normalize_spr(stack / pot)
    facing_raise = random.randint(0, 1)
    bettings = np.array([
        position,
        spr,
        facing_raise,
        agg / 4,
        agg_opp / 4,
    ], dtype=np.float32)
    hole_cards, community_cards = generate_cards()
    return (
        convert_hand(hole_cards, 2),
        convert_hand(community_cards, 5),
        np.int32(street_idx),
        bettings
    )

def compute_strategy(strategy_net: DeepCFRModel, infoset: tuple) -> np.ndarray:
    strategy_net.eval()
    hole_cards, community_cards, street, bettings = infoset
    with torch.no_grad():
        logits = strategy_net(
            torch.tensor(np.expand_dims(hole_cards, axis = 0), device=DEVICE),
            torch.tensor(np.expand_dims(community_cards, axis = 0), device=DEVICE),
            torch.tensor(np.expand_dims(street, axis = 0), device=DEVICE),
            torch.tensor(np.expand_dims(bettings, axis = 0), device=DEVICE)
        )
        strategy = F.softmax(logits, dim=1).squeeze()
    strategy_net.train()
    return strategy.detach().cpu().numpy()


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.seen = Bloom(500_000_000, 0.01)

    def __len__(self):
        return len(self.buffer)

    def insert(self, item):
        hashable_item = make_hashable(item)
        if hashable_item not in self.seen:
            self.buffer.append(item)
            self.seen.add(hashable_item)

    def sample(self, k: int) -> List[tuple]:
        return random.sample(self.buffer, k)

    def extend(self, other: Self):
        self.buffer.extend(other.buffer)
        self.seen = self.seen | other.seen

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)

    @staticmethod
    def load(filepath: str) -> Self:
        with open(filepath, 'rb') as f:
            buffer = pickle.load(f)
        rb = ReplayBuffer()
        rb.buffer = buffer
        return rb

    def __deepcopy__(self, memo):
        new_instance = ReplayBuffer()
        new_instance.seen = self.seen.copy()
        # we do not need the buffer to deepcopied only the seen
        # new_instance.buffer = deepcopy(self.buffer, memo)
        return new_instance

class DeepCFRAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.advantage_net = DeepCFRModel().to(DEVICE)
        self.advantage_memory = ReplayBuffer()

    def get_avg_action(self, valid_actions: List[str], infoset: tuple) -> np.ndarray:
        valid_action_indices = [ACTIONS.index(action) for action in valid_actions]
        strategy = compute_strategy(self.advantage_net, infoset)
        action_mask = np.zeros(ACTION_SPACE)
        action_mask[valid_action_indices] = 1.0
        masked_strategy = strategy * action_mask
        return masked_strategy
    
    def get_best_action(self, valid_actions: List[str], infoset: torch.Tensor) -> int:
        return self.get_avg_action(valid_actions, infoset).argmax()
    
    def reinit_net(self):
        self.advantage_net = DeepCFRModel().to(DEVICE)

    def __deepcopy__(self, memo):
        new_instance = DeepCFRAgent(self.player_id)
        cpu_model = self.advantage_net.to('cpu')
        new_instance.advantage_net = deepcopy(cpu_model, memo)
        new_instance.advantage_memory = deepcopy(self.advantage_memory, memo)
        self.advantage_net.to(DEVICE)
        new_instance.advantage_net.to(DEVICE)
        return new_instance
    
def get_shuffled_batches(buffer: list, batch_size: int) -> Generator[list, None, None]:
    indices = list(range(len(buffer)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield [buffer[j] for j in batch_indices]

def get_batches(buffer: list, batch_size: int) -> Generator[list, None, None]:
    for i in range(0, len(buffer), batch_size):
        yield buffer[i:i + batch_size]

def prepare_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hole_cards_batch = []
    community_cards_batch = []
    street_batch = []
    bettings_batch = []
    regrets_batch = []
    t_prime_batch = []
    for info, regrets, t_prime in batch:
        hole_cards, community_cards, street, bettings = info
        hole_cards_batch.append(hole_cards)
        community_cards_batch.append(community_cards)
        street_batch.append(street)
        bettings_batch.append(bettings)
        regrets_batch.append(regrets)
        t_prime_batch.append(t_prime)

    return (
        torch.tensor(np.array(hole_cards_batch), device=DEVICE),
        torch.tensor(np.array(community_cards_batch), device=DEVICE),
        torch.tensor(np.array(street_batch), device=DEVICE),
        torch.tensor(np.array(bettings_batch), device=DEVICE),
        torch.tensor(np.array(regrets_batch), device=DEVICE),
        torch.tensor(np.array(t_prime_batch), device=DEVICE),
    )

def compute_validation_loss(net, val_buffer, batch_size):
    net.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    with torch.no_grad():
        for batch in get_batches(val_buffer, batch_size):
            hole_cards_tensor, community_cards_tensor, street_tensor, bettings_tensor, regrets_tensor, t_prime = prepare_batch(batch)
            action_advantages = net(hole_cards_tensor, community_cards_tensor, street_tensor, bettings_tensor)
            squared_diff = (action_advantages - regrets_tensor) ** 2
            sum_over_actions = squared_diff.sum(dim=1)
            weighted_sum = (t_prime + 1) * sum_over_actions
            loss = weighted_sum.mean()
            total_val_loss += loss.item()
            num_val_batches += 1
    net.train()
    return total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

def train_network(
        net: DeepCFRModel,
        memory: ReplayBuffer,
        lr: float = LR,
        batch_size: int = TRAIN_BATCH_SIZE,
        epochs: int = TRAIN_EPOCHS,
        filename: Optional[str] = None,
        log: bool = True
    ) -> float:
    net.to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Training/validation split
    val_split = 0.2
    indices = list(range(len(memory)))
    random.shuffle(indices)
    train_size = int((1 - val_split) * len(indices))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_buffer = [memory.buffer[i] for i in train_indices]
    val_buffer = [memory.buffer[i] for i in val_indices]
    
    best_val_loss = float('inf')
    patience = TRAINING_PATIENCE
    no_improve = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training phase
        total_train_loss = 0.0
        num_train_batches = 0
        for batch in get_shuffled_batches(train_buffer, batch_size):
            optimizer.zero_grad()
            hole_cards_tensor, community_cards_tensor, street_tensor, bettings_tensor, regrets_tensor, t_prime = prepare_batch(batch)
            action_advantages = net(hole_cards_tensor, community_cards_tensor, street_tensor, bettings_tensor)
            squared_diff = (action_advantages - regrets_tensor) ** 2
            sum_over_actions = squared_diff.sum(dim=1)
            weighted_sum = (t_prime + 1) * sum_over_actions
            loss = weighted_sum.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            num_train_batches += 1
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        
        # Validation phase
        val_loss = compute_validation_loss(net, val_buffer, batch_size)
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_epoch = epoch
            if filename is not None:
                torch.save(net.state_dict(), filename)
        else:
            no_improve += 1
        
        if no_improve >= patience:
            if log: print(f"Early stopping at epoch {best_epoch}")
            break
        
        # Step the learning rate scheduler
        scheduler.step()
        if log and epoch % 50 == 0: print(f"Epoch [{epoch+1}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load the best model
    if filename is not None:
        net.load_state_dict(torch.load(filename))
    
    return best_val_loss

def traverse(
    emulator: Emulator,
    game_state: dict,
    events: dict,
    agent: DeepCFRAgent,
    opponent: DeepCFRAgent,
    strategy_memory: ReplayBuffer,
    iteration: int
) -> float:
    traversing_player_idx = agent.player_id
    round_state: dict = events[-1]['round_state']
    if _is_terminal(game_state):
        return _get_payoff(round_state, traversing_player_idx)

    current_player_idx: int = game_state['next_player']
    infoset = _get_infoset(current_player_idx, game_state, round_state)
    valid_actions = list(map(extract_action, events[-1]['valid_actions']))

    if current_player_idx == traversing_player_idx:
        strategy = compute_strategy(agent.advantage_net, infoset)
        v_a = np.zeros(3, dtype=np.float32)

        for action in valid_actions:
            new_game_state, new_events = _apply_action(emulator, game_state, action)
            idx = ACTIONS.index(action)
            v_a[idx] = traverse(
                emulator,
                new_game_state,
                new_events,
                agent,
                opponent,
                strategy_memory,
                iteration,
            )
        v = sum(strategy[a] * v_a[a] for a in range(ACTION_SPACE))
        advantages = [v_a[a] - v for a in range(ACTION_SPACE)]
        if random.random() > 0.1:
            agent.advantage_memory.insert((infoset, advantages, iteration))
        return v
    else:
        strategy = compute_strategy(opponent.advantage_net, infoset)
        if random.random() > 0.1:
            strategy_memory.insert((infoset, strategy, iteration))

        probs = strategy / strategy.sum()
        action = np.random.choice(ACTIONS, p=probs)
        
        new_game_state, new_events = _apply_action(emulator, game_state, action)

        return traverse(
            emulator,
            new_game_state,
            new_events,
            agent,
            opponent,
            strategy_memory,
            iteration,
        )

def _run_traversals_for_player(inner_agents, player, nb_game_tree_traversals, emulator, game_state, events, iteration_t):
    inner_strategy_memory = ReplayBuffer()

    for _ in range(nb_game_tree_traversals):
        traverse(
            emulator,
            game_state,
            events,
            inner_agents[player],
            inner_agents[1 - player],
            inner_strategy_memory,
            iteration_t,
        )

    return inner_agents, inner_strategy_memory, player


def deep_cfr(
    nb_iterations: int = TRAIN_ITERATIONS,
    nb_players: int = 2,
    nb_game_tree_traversals: int = GAME_TRAVERSALS_PER_HAND,
    workers: Optional[int] = None,
    log_every: int = 5
):
    if workers is None:
        workers = os.cpu_count()
    print(f"Training CFR agents using {workers} workers")

    agents = [DeepCFRAgent(i) for i in range(nb_players)]
    strategy_memory = ReplayBuffer()

    for iteration_t in tqdm.tqdm(range(nb_iterations)):
        log = log_every != 0 and iteration_t % log_every == 0 and iteration_t > 0
        if log: print("Iteration: ", iteration_t)

        emulator, game_state, events = get_initial_state()

        for player in range(nb_players):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []

                for i in range(workers):
                    futures.append(
                        executor.submit(
                            _run_traversals_for_player,
                            [deepcopy(agent) for agent in agents],
                            player,
                            nb_game_tree_traversals // workers,
                            deepcopy(emulator),
                            deepcopy(game_state),
                            deepcopy(events),
                            iteration_t
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    inner_agents, inner_strategy_memory, inner_player = future.result()
                    strategy_memory.extend(inner_strategy_memory)
                    agents[inner_player].advantage_memory.extend(inner_agents[inner_player].advantage_memory)

            # Train advantage network
            agents[player].reinit_net()
            train_network(
                net=agents[player].advantage_net,
                memory=agents[player].advantage_memory,
                filename='./models/advantage_network.pth'
            )

            if log:
                if player == 1:
                    print(f'\nTop hand strategy: {evaluate_hole_cards(agents[player], ["SA", "CA"])}')
                    print(f'Worst hand strategy: {evaluate_hole_cards(agents[player], ["H2", "C7"])}')
                print(f'Overall strategy for player {player}: {evaluate_average_strategy(agents[player])}')

                if iteration_t > 0 and iteration_t % 10 == 0:
                    wr = train_strategy_and_evaluate(strategy_memory)
                    print(f'WR against call player: {wr}')

    print('Training final CFR model...')
    strategy_net = DeepCFRModel()
    train_network(
        net=strategy_net,
        memory=strategy_memory,
        filename='./models/strategy.pth'
    )
    return strategy_net

def train_strategy_and_evaluate(strategy_memory):
    strategy_net = DeepCFRModel().to(DEVICE)
    train_network(
        net=strategy_net,
        memory=strategy_memory,
        filename='./models/strategy.pth'
    )
    wr = run_n_games(CFRPokerPlayer(strategy_net), 'cfr', CallPlayer(), 'call_player', 1000)
    return wr

def evaluate_hole_cards(agent: DeepCFRAgent, hole_cards: List[str] = ['SA', 'CA']) -> dict:
    _, game_state, events = get_initial_state_with_cards(hole_cards, f'uuid-{1}')
    round_state: dict = events[-1]['round_state']
    current_player_idx: int = game_state['next_player']
    assert(current_player_idx == 1)
    infoset = _get_infoset(current_player_idx, game_state, round_state)
    valid_actions = list(map(extract_action, events[-1]['valid_actions']))
    avg = agent.get_avg_action(valid_actions, infoset)
    return {ACTIONS[i]: avg[i] for i in range(len(ACTIONS))}

def evaluate_average_strategy(agent: DeepCFRAgent, samples: int = 1000) -> dict:
    avg = np.zeros(3, dtype=np.float32)
    for i in range(samples):
        infoset = fake_info_set(i % 2)
        avg += agent.get_avg_action(ACTIONS, infoset)
    return avg / samples