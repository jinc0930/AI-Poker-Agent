## commands used to run
pyenv activate ai-poker-agent
pip install torch
pip install numpy
pip install tensorboard

    python3 ./PPO/train.py  
    tensorboard --logdir ./PPO/runs 

## references 
papers (PPO):
https://arxiv.org/pdf/1707.06347
https://openai.com/index/openai-baselines-ppo/

Other related papers:
CFR - https://youtu.be/MWRXx2saLw4?si=JC6D00MwWRggIowH & https://github.com/Gongsta/Poker-AI/
CFR - https://www.youtube.com/watch?v=nSrbai9kIeA&t=423s & https://www.youtube.com/watch?v=NWS9v_r_IWk

code/inspirations:
https://github.com/ishikota/PyPokerEngine
https://github.com/datvodinh/ppo-transformer
https://github.com/curvysquare/PPO-and-A2C-for-HULH-poker
https://github.com/seungeunrho/minimalRL/tree/master

## Notes/Questions
BB/100 metric

TODO:
Need to update trained metrics and keep it saved somehow
Need to think of opponent features - currently not really learning from opponent behaviors
Should we throw in wild card rounds? (i.e. decide strategy with some uncertainty)