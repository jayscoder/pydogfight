python main.py scripts/thesis/ppo-A_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-B_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-C_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-D_vs_greedy.yaml --train


python main.py scripts/thesis/ppo-A-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-B-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-C-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis/ppo-D-ar5-d400_vs_greedy.yaml --train



tensorboard --logdir=scripts/thesis
cd /Users/wangtong/Documents/GitHub/pydogfight
