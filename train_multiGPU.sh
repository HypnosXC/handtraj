source /root/miniconda3/bin/activate handgen
accelerate launch --num_processes 8 --num_machines 1 train_hand_motion_prior.py --config.experiment-name interhand_fixed --config.batch_size 2048 