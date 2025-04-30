## Term Project

### Set up environment
Libraries:
numpy==2.0.2
torch==2.6.0

### Latest Version
./PPO_REDONE/submission

custom_player = PPO player
all other files in the 'submission' folder is needed to run it

To test your agent:
uncomment in custom_player.py

```python3
# uncomment to run against other players
# if __name__ == "__main__":
#     wr = run_n_games(setup_ai(), 'AI', CallPlayer(), 'AI2')
#     print(f'yippie{wr}')
```
change "CallPlayer()" to your player
wr shown is PPO winrate against the target player
