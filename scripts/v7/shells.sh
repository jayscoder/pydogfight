python main.py scripts/v7/sac-A-shared_vs_greedy.yaml --train
python main.py scripts/v7/sac-A_vs_greedy.yaml --train
python main.py scripts/v7/greedy_vs_greedy.yaml --train
python main.py scripts/v7

tensorboard --logdir=output
