python main.py scripts/v9/ppo-A_vs_greedy.yaml --train
python main.py scripts/v9/ppo-B_vs_greedy.yaml --train
python main.py scripts/v9/ppo-C_vs_greedy.yaml --train
python main.py scripts/v9/ppo-D_vs_greedy.yaml --train
python main.py scripts/v9/ppo-E_vs_greedy.yaml --train

python main.py scripts/v9/ppo-C_vs_greedy.yaml scripts/v9/ppo-D_vs_greedy.yaml scripts/v9/ppo-E_vs_greedy.yaml --train

tensorboard --logdir=scripts/v9

cd /Users/wangtong/Documents/GitHub/pydogfight
