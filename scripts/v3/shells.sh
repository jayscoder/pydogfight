python main.py scripts/v3/run.yaml --rl-algo=ppo --output-dir=output/v3/ppo --models-dir=models/v3/ppo --train
python main.py scripts/v3/run.yaml --rl-algo=sac --output-dir=models/v3/sac --models-dir=models/v3/sac --train


python main.py scripts/v3/run.yaml --rl-algo=ppo --output-dir=output/v3/ppo_L --models-dir=models/v3/ppo_L --num-episodes=10000 --train
python main.py scripts/v3/run.yaml --rl-algo=sac --output-dir=output/v3/sac_L --models-dir=models/v3/sac_L --num-episodes=10000 --train
