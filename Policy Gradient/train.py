import sys
import time
import os
import random
import json
import numpy as np
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from call_player import CallOnlyPlayer
from fold_player import FoldOnlyPlayer
from custom_player import CustomPlayer
from MonteCarlo.montecarlo_player import MonteCarloPlayer

def linear_schedule(initial_value, final_value, current_step, total_steps):
    if current_step >= total_steps:
        return final_value
    fraction = current_step / total_steps
    return initial_value + fraction * (final_value - initial_value)

def create_random_weights():
    weights = {
        'hand_strength': random.uniform(0.1, 0.6),
        'pot_odds': random.uniform(0.1, 0.5),
        'position': random.uniform(0.1, 0.4),
        'aggression': random.uniform(0.05, 0.3)
    }
    
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
        
    return weights

def train_against_opponents(iterations=100, use_random_weights=False, use_json=True):
    print("Starting training session")
    print(f"Training for {iterations} iterations")
    print(f"Random weights initialization: {'Enabled' if use_random_weights else 'Disabled'}")
    print(f"Using JSON format: {'Enabled' if use_json else 'Disabled'}")
    
    file_ext = '.json' if use_json else '.pkl'
    weights_file = f'player_weights{file_ext}'
    
    if os.path.exists(weights_file):
        os.remove(weights_file)
        print(f"Deleted existing weights file '{weights_file}' to start fresh")
    
    max_round = 10
    initial_stack = 1000
    small_blind_amount = 5
    
    training_agent = CustomPlayer(is_training=True, learning_rate=0.05, 
                                 discount_factor=0.9, exploration_rate=0.4,
                                 weights_file=weights_file, use_json=use_json)
    
    if use_random_weights:
        random_weights = create_random_weights()
        training_agent.weights = random_weights
        print(f"Initial random weights: {training_agent.weights}")
    else:
        print(f"Initial default weights: {training_agent.weights}")
    
    opponent_players = {
        "Random": RandomPlayer(),
        "Raised": RaisedPlayer(),
        "CallOnly": CallOnlyPlayer(),
        "FoldOnly": FoldOnlyPlayer(),
        "MonteCarlo-Easy": MonteCarloPlayer(difficulty=0.3),
        "MonteCarlo-Medium": MonteCarloPlayer(difficulty=0.5),
        "MonteCarlo-Hard": MonteCarloPlayer(difficulty=0.8)
    }
    
    total_games = iterations * len(opponent_players)
    wins = 0
    total_profit = 0
    
    print(f"Starting training against {len(opponent_players)} opponents...")
    start_time = time.time()
    
    for iteration in range(iterations):
        for opponent_name, opponent in opponent_players.items():
            if iteration % 2 == 0:
                config = setup_config(max_round=max_round, 
                                     initial_stack=initial_stack, 
                                     small_blind_amount=small_blind_amount)
                config.register_player(name="TrainingAgent", algorithm=training_agent)
                config.register_player(name="Opponent", algorithm=opponent)
            else:
                config = setup_config(max_round=max_round, 
                                     initial_stack=initial_stack, 
                                     small_blind_amount=small_blind_amount)
                config.register_player(name="Opponent", algorithm=opponent)
                config.register_player(name="TrainingAgent", algorithm=training_agent)
            
            game_result = start_poker(config, verbose=0)
            
            our_stack = 0
            for player in game_result['players']:
                if player['name'] == "TrainingAgent":
                    our_stack = player['stack']
                    break
            
            win = our_stack > initial_stack
            if win:
                wins += 1
            
            profit = our_stack - initial_stack
            total_profit += profit
            
            training_agent.exploration_rate = linear_schedule(
                0.4, 0.1, iteration * len(opponent_players), total_games
            )
            
            games_played = iteration * len(opponent_players) + list(opponent_players.keys()).index(opponent_name) + 1
            if games_played % 100 == 0 or games_played == total_games:
                training_agent._save_weights()
                win_rate = wins / games_played
                training_agent._save_weights(f"player_weights_{win_rate:.2f}{file_ext}")
                print(f"\nSaved weights at game {games_played}/{total_games} - Win rate: {win_rate:.2f}")
            
            progress = games_played / total_games * 100
            if games_played % 10 == 0 or games_played == total_games:
                avg_profit = total_profit / games_played
                win_rate = wins / games_played
                duration = time.time() - start_time
                print(f"\rProgress: {progress:.1f}% - Games: {games_played}/{total_games} - Win rate: {win_rate:.2f} - Avg profit: {avg_profit:.1f} - Time: {duration:.0f}s", end="")
    
    training_agent._save_weights()
    
    final_win_rate = wins / total_games
    if final_win_rate > 0.55:
        best_file = f"player_weightsBest{file_ext}"
        training_agent._save_weights(best_file)
        print(f"\nSaved weights as '{best_file}' (Win rate: {final_win_rate:.2f})")
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f} seconds")
    print(f"Final statistics - Games: {total_games}, Wins: {wins}, Win rate: {final_win_rate:.2f}, Avg profit: {total_profit/total_games:.1f}")
    print(f"Final weights: {training_agent.weights}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train the poker AI agent')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations to train')
    parser.add_argument('--random-weights', action='store_true',
                       help='Start with random weights instead of defaults')
    parser.add_argument('--use-pickle', action='store_true',
                       help='Use pickle format instead of JSON for weights')
    
    args = parser.parse_args()
    
    train_against_opponents(iterations=args.iterations, 
                           use_random_weights=args.random_weights,
                           use_json=not args.use_pickle)