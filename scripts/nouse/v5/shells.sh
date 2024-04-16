python main.py scripts/v5/run.yaml --rl-algo=ppo --output-dir=output/v5/ppo --models-dir=models/v5/ppo --train --render
python main.py scripts/v5/run.yaml --rl-algo=sac --output-dir=output/v5/sac --models-dir=models/v5/sac --train

python main.py scripts/v5/run.yaml --rl-algo=ppo --output-dir=output/v5/ppo --models-dir=models/v5/ppo --num-episodes=100
python main.py scripts/v5/run.yaml --rl-algo=ppo --output-dir=output/v5/sac --models-dir=models/v5/sac --num-episodes=100

python main.py scripts/v5/run.yaml --rl-algo=ppo --output-dir=output/v5/ppo_L --models-dir=models/v5/ppo_L --num-episodes=10000 --train
python main.py scripts/v5/run.yaml --rl-algo=sac --output-dir=output/v5/sac_L --models-dir=models/v5/sac_L --num-episodes=10000 --train
