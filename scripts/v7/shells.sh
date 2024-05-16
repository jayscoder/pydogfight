python main.py scripts/v7/sac-A_vs_greedy.yaml scripts/v7/sac-A-shared_vs_greedy.yaml --train
python main.py scripts/v7/sac-B-f32_vs_greedy.yaml scripts/v7/sac-B-f64_vs_greedy.yaml scripts/v7/sac-B-f128_vs_greedy.yaml --train

python main.py scripts/v7/sac-B-f32_vs_greedy.yaml --train

python main.py scripts/v7/sac-B-nof_vs_greedy.yaml --train

python main.py scripts/v7/sac-C-nof_vs_greedy.yaml --train

# TORUN

python main.py scripts/v7/sac-B-f32_vs_greedy.yaml scripts/v7/sac-B-lstm-f32_vs_greedy.yaml --train

python main.py scripts/v7/sac-B-f64_vs_greedy.yaml scripts/v7/sac-B-f128_vs_greedy.yaml --train

python main.py scripts/v7/sac-A_vs_sac-A-shared.yaml --train

tensorboard --logdir=scripts/v7


