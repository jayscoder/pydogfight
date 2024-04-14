python main.py scripts/v4/run.yaml --rl-algo=ppo --output-dir=output/v4/ppo --models-dir=models/v4/ppo --train --render
python main.py scripts/v4/run.yaml --rl-algo=sac --output-dir=output/v4/sac --models-dir=models/v4/sac --train

python main.py scripts/v4/run.yaml --rl-algo=ppo --output-dir=output/v4/ppo --models-dir=models/v4/ppo --num-episodes=100
python main.py scripts/v4/run.yaml --rl-algo=ppo --output-dir=output/v4/sac --models-dir=models/v4/sac --num-episodes=100


python main.py scripts/v4/greedy.yaml

python main.py scripts/v4/rl-greedy.yaml --rl-algo=ppo --output-dir=output/v4/ppo-greedy --models-dir=models/v4/ppo-greedy --train
python main.py scripts/v4/rl-greedy-10000.yaml --rl-algo=ppo --output-dir=output/v4/ppo-greedy-10000 --models-dir=models/v4/ppo-greedy-10000 --train

python main.py scripts/v4/rl-greedy.yaml --rl-algo=sac --output-dir=output/v4/sac-greedy --models-dir=models/v4/ppo

python main.py scripts/v4/greedy.yaml --output-dir=output/v4/greedy
python main.py scripts/v4/greedy.yaml --output-dir=output/v4/greedy


python main.py scripts/v4/run.yaml --rl-algo=ppo --output-dir=output/v4/ppo_L --models-dir=models/v4/ppo_L --num-episodes=10000 --train
python main.py scripts/v4/run.yaml --rl-algo=sac --output-dir=output/v4/sac_L --models-dir=models/v4/sac_L --num-episodes=10000 --train
