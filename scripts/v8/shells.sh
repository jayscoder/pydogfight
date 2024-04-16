python train.py scripts/v6/ppo-greedy.yaml
python train.py scripts/v6/ppo-greedy-2.yaml

python train.py scripts/v6/sac-greedy.yaml
python train.py scripts/v6/sac-greedy-2.yaml

python evaluate.py scripts/v6/ppo-greedy.yaml --render
python evaluate.py scripts/v6/ppo-greedy-2.yaml --render

python evaluate.py scripts/v6/sac-greedy.yaml --render
python evaluate.py scripts/v6/sac-greedy-2.yaml --render



pybts --dir=output/v6

tensorboard --logdir=output
