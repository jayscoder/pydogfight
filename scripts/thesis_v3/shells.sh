python main.py scripts/thesis_v3/ppo-A.yaml --train
python main.py scripts/thesis_v3/ppo-B.yaml --train
python main.py scripts/thesis_v3/ppo-C.yaml --train
python main.py scripts/thesis_v3/ppo-D.yaml --train
python main.py scripts/thesis_v3/ppo-E.yaml --train
python main.py scripts/thesis_v3/ppo-F.yaml --train
python main.py scripts/thesis_v3/ppo-G.yaml --train
python main.py scripts/thesis_v3/ppo-I.yaml --train
python main.py scripts/thesis_v3/ppo-J.yaml --train

python main.py scripts/thesis_v3/ppo-A-ar.yaml --train
python main.py scripts/thesis_v3/ppo-B-ar.yaml --train
python main.py scripts/thesis_v3/ppo-C-ar.yaml --train
python main.py scripts/thesis_v3/ppo-D-ar.yaml --train
python main.py scripts/thesis_v3/ppo-E-ar.yaml --train
python main.py scripts/thesis_v3/ppo-F-ar.yaml --train
python main.py scripts/thesis_v3/ppo-G-ar.yaml --train
python main.py scripts/thesis_v3/ppo-H-ar.yaml --train

python main.py scripts/thesis_v3/ppo-I-ar.yaml --train
python main.py scripts/thesis_v3/ppo-J-ar.yaml --train

tensorboard --logdir=scripts/thesis_v3
cd /Users/wangtong/Documents/GitHub/pydogfight
