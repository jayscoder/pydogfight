python main.py scripts/v6/ppo-greedy.yaml --train
python main.py scripts/v6/ppo-greedy-2.yaml

python main.py scripts/v6/sac-greedy.yaml --train
python main.py scripts/v6/sac-greedy-2.yaml --train

python main.py scripts/v6/ppo-greedy.yaml --render
python main.py scripts/v6/ppo-greedy-2.yaml --render

python main.py scripts/v6/sac-greedy.yaml --render
python main.py scripts/v6/sac-greedy-2.yaml --render



pybts --dir=output/v6

tensorboard --logdir=output
