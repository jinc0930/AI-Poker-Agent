from hand_strenth import generate_training_data
from players import CFRPokerPlayer, MonteCarloPlayer, RandomPlayer
from utils import run_n_games

if __name__ == "__main__":
    print("Evaluating CFR agent...")
    cfr_agent = CFRPokerPlayer.load_from_file('./checkpoints/cfr_checkpoint_5000.pickle')

    result = run_n_games(cfr_agent, 'CFR', RandomPlayer(), 'RandomPlayer', n_games=100)
    print(f'WR against RandomPlayer: {result}')

    result = run_n_games(cfr_agent, 'CFR', MonteCarloPlayer(1), 'MonteCarlo', n_games=100)
    print(f'WR against MonteCarloPlayer: {result}')
