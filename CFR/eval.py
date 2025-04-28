from players import CFRPokerPlayer, CallPlayer, FoldPlayer, MonteCarloPlayer, RandomPlayer
from utils import run_n_games

if __name__ == "__main__":
    print("Evaluating CFR agent...")
    cfr_agent = CFRPokerPlayer()

    result = run_n_games(cfr_agent, 'CFR', FoldPlayer(), 'FoldPlayer', n_games=100)
    print(f'WR against FoldPlayer: {result}')

    result = run_n_games(cfr_agent, 'CFR', RandomPlayer(), 'RandomPlayer', n_games=100)
    print(f'WR against RandomPlayer: {result}')

    result = run_n_games(cfr_agent, 'CFR', CallPlayer(), 'CallPlayer', n_games=100)
    print(f'WR against CallPlayer: {result}')

    result = run_n_games(cfr_agent, 'CFR', MonteCarloPlayer(0.5), 'MonteCarloPlayer', n_games=100)
    print(f'WR against MonteCarloPlayer: {result}')
