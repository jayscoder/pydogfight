python main.py scripts/v6/ppo-greedy.yaml --train --render
python main.py scripts/v6/ppo-greedy-2.yaml --train

python main.py scripts/v6/sac-greedy.yaml --train --render
python main.py scripts/v6/sac-greedy-2.yaml --train




pybts --dir=output/v6

tensorboard --logdir=output
