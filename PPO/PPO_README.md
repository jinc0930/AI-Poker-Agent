## commands used to run
pyenv activate ai-poker-agent
pip install torch
pip install numpy
pip install tensorboard
pip install phevaluator

    python3 ./PPO/train.py  
    tensorboard --logdir ./PPO/runs 

## Human testing against poker agent
pip install pypokergui
pypokergui build_config --maxround 500 --stack 1000 --small_blind 10 --ante 0 >> PPO/poker_conf.yaml
pypokergui serve /Users/changjin/baa-baa-barn/AI-Poker-Agent/PPO/poker_conf.yaml --port 8000 --speed moderate
    ## MACos need HHTP 
        Safari > Settings > Security > Disable HTTP warning
    ## bug fix 
        https://github.com/ishikota/PyPokerGUI/issues/6#event-1254261637


## references 
Papers (PPO):
https://arxiv.org/pdf/1707.06347
https://openai.com/index/openai-baselines-ppo/
Websites:
https://www.raketherake.com/news/2023/05/win-percentage-of-every-poker-starting-hands

Other related papers:
CFR - https://youtu.be/MWRXx2saLw4?si=JC6D00MwWRggIowH & https://github.com/Gongsta/Poker-AI/
CFR - https://www.youtube.com/watch?v=nSrbai9kIeA&t=423s & https://www.youtube.com/watch?v=NWS9v_r_IWk

code/inspirations:
https://github.com/ishikota/PyPokerEngine
https://github.com/datvodinh/ppo-transformer
https://github.com/curvysquare/PPO-and-A2C-for-HULH-poker
https://github.com/seungeunrho/minimalRL/tree/master
