python main.py scripts/v8/sac-A_vs_greedy.yaml --train
python main.py scripts/v8/greedy_vs_greedy.yaml --train
python main.py scripts/v8/sac-B_vs_greedy.yaml --train
python main.py scripts/v8/sac-C_vs_greedy.yaml --train
python main.py scripts/v8/sac-A_vs_greedy.yaml scripts/v8/sac-A-shared_vs_greedy.yaml scripts/v8/sac-A_vs_sac-A-shared.yaml --train
python main.py scripts/v8/sac-D_vs_greedy.yaml --train
tensorboard --logdir=scripts/v8/output


python main.py scripts/v8/sac-B_vs_greedy.yaml --render
python main.py scripts/v8/greedy_vs_greedy2.yaml --render

python main.py scripts/v8/ppo-A_vs_greedy.yaml --train
python main.py scripts/v8/ppo-B_vs_greedy.yaml --train

python main.py scripts/v8/ppo-C_vs_greedy.yaml --train
python main.py scripts/v8/ppo-E_vs_greedy.yaml --train

python main.py scripts/v8/ppo-B_vs_ppo-B.yaml --train

python main.py scripts/v8/sac-B_vs_greedy.yaml --train
python main.py scripts/v8/sac-E_vs_greedy.yaml --train

python main.py scripts/v8/ppo-B_vs_greedy.yaml --render
python main.py scripts/v8/ppo-E_vs_greedy.yaml --render

python main.py scripts/v8/ppo-B_vs_manual.yaml --render

python main.py scripts/v8/greedy_A_vs_greedy_B.yaml --render

python main.py scripts/v8/ppo-A_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-B_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-C_vs_greedy_evade.yaml --train

python main.py scripts/v8/ppo-B_ap_5_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-B_ap_5_decay_vs_greedy_evade.yaml --train

python main.py scripts/v8/ppo-B_reward_B_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-B_ap_5_reward_B_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-B_ap_5_decay_400_reward_B_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-B_ap_5_clamp_reward_B_vs_greedy_evade.yaml --train

python main.py scripts/v8/ppo-A_ap_5_decay_400_reward_B_vs_greedy_evade.yaml --train
python main.py scripts/v8/ppo-C_ap_5_decay_400_reward_B_vs_greedy_evade.yaml --train

ppo_B/2
sac_B/13
sac_E/7
ppo_E/8

ppo_E/12 规避奖励1
ppo_B/16 规避奖励1

ppo_E/13 规避奖励0.2
ppo_B/18 规避奖励0.2


cd /Users/wangtong/Documents/GitHub/pydogfight
