python main.py scripts/thesis_evade/ppo-A_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-B_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-C_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-D_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-E_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-F_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-G_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-I_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-J_vs_greedy.yaml --train

python main.py scripts/thesis_evade/ppo-A-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-B-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-C-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-D-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-E-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-F-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-G-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-H-ar5-d400_vs_greedy.yaml --train

python main.py scripts/thesis_evade/ppo-I-ar5-d400_vs_greedy.yaml --train
python main.py scripts/thesis_evade/ppo-J-ar5-d400_vs_greedy.yaml --train

tensorboard --logdir=scripts/thesis_evade
cd /Users/wangtong/Documents/GitHub/pydogfight
