import os
import json
import time
import numpy as np
from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from call_player import CallOnlyPlayer
from fold_player import FoldOnlyPlayer
from custom_player import CustomPlayer
from MonteCarlo.montecarlo_player import MonteCarloPlayer

def test_against_opponents(weights_file='player_weights.json', use_json=True):
    if not os.path.exists(weights_file):
        if use_json and os.path.exists(weights_file.replace('.json', '.pkl')):
            weights_file = weights_file.replace('.json', '.pkl')
            use_json = False
            print(f"JSON weights file not found. Using pickle file: {weights_file}")
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    print(f"Testing agent using weights from: {weights_file}")
    print(f"Format: {'JSON' if use_json else 'Pickle'}")
    
    our_player = CustomPlayer(weights_file=weights_file, is_training=False, 
                             exploration_rate=0.0, use_json=use_json)
    
    test_games = 50
    max_round = 10
    initial_stack = 1000
    small_blind_amount = 5
    
    opponents = {
        "Random": RandomPlayer(),
        "Raised": RaisedPlayer(),
        "CallOnly": CallOnlyPlayer(),
        "FoldOnly": FoldOnlyPlayer(),
        "MonteCarlo-Easy": MonteCarloPlayer(difficulty=0.3),
        "MonteCarlo-Medium": MonteCarloPlayer(difficulty=0.5),
        "MonteCarlo-Hard": MonteCarloPlayer(difficulty=0.8)
    }
    
    overall_results = []
    opponent_stats = {}
    
    start_time = time.time()
    
    for opponent_name, opponent in opponents.items():
        print(f"\nTesting against {opponent_name}...")
        wins = 0
        total_profit = 0
        
        for game_num in range(test_games):
            if game_num % 2 == 0:
                config = setup_config(max_round=max_round, 
                                     initial_stack=initial_stack, 
                                     small_blind_amount=small_blind_amount)
                config.register_player(name="OurPlayer", algorithm=our_player)
                config.register_player(name="Opponent", algorithm=opponent)
            else:
                config = setup_config(max_round=max_round, 
                                     initial_stack=initial_stack, 
                                     small_blind_amount=small_blind_amount)
                config.register_player(name="Opponent", algorithm=opponent)
                config.register_player(name="OurPlayer", algorithm=our_player)
            
            game_result = start_poker(config, verbose=0)
            
            our_stack = 0
            opponent_stack = 0
            
            for player in game_result['players']:
                if player['name'] == "OurPlayer":
                    our_stack = player['stack']
                else:
                    opponent_stack = player['stack']
            
            win = our_stack > opponent_stack
            if win:
                wins += 1
                
            profit = our_stack - initial_stack
            total_profit += profit
            
            progress = (game_num + 1) / test_games * 100
            print(f"\rProgress: {progress:.1f}% - Wins: {wins}/{game_num+1} ({wins/(game_num+1)*100:.1f}%)", end="")
        
        win_rate = wins / test_games
        avg_profit = total_profit / test_games
        opponent_stats[opponent_name] = {
            'win_rate': win_rate,
            'avg_profit': avg_profit
        }
        overall_results.append(win_rate)
        
        print(f"\rResults vs {opponent_name}: Win rate: {win_rate:.2f} ({wins}/{test_games}) - Avg profit: {avg_profit:.1f}")
    
    duration = time.time() - start_time
    overall_win_rate = sum(overall_results) / len(overall_results)
    
    print("\n" + "="*50)
    print(f"Overall Results ({len(opponents)} opponents):")
    print(f"Average win rate: {overall_win_rate:.2f}")
    print(f"Test duration: {duration:.2f} seconds")
    print("="*50)
    print("\nDetailed results by opponent:")
    
    sorted_opponents = sorted(opponent_stats.items(), 
                             key=lambda x: x[1]['win_rate'], 
                             reverse=True)
    
    for opponent_name, stats in sorted_opponents:
        print(f"{opponent_name:15}: Win rate: {stats['win_rate']:.2f} - Avg profit: {stats['avg_profit']:.1f}")
    
    print("\nAgent weights:")
    print(our_player.weights)
    
    return overall_win_rate, opponent_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test the poker AI agent')
    parser.add_argument('--weights-file', type=str, default='player_weights.json',
                       help='Path to the weights file')
    parser.add_argument('--use-pickle', action='store_true',
                       help='Use pickle format instead of JSON for weights')
    
    args = parser.parse_args()
    
    if args.use_pickle and not args.weights_file.endswith('.pkl'):
        args.weights_file = args.weights_file.replace('.json', '.pkl')
    elif not args.use_pickle and not args.weights_file.endswith('.json'):
        args.weights_file = args.weights_file.replace('.pkl', '.json')
    
    test_against_opponents(weights_file=args.weights_file, use_json=not args.use_pickle)