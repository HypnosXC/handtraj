#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=handgen
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

# ---- 在这里加载环境 ----
source activate handtraj

# ---- 在这里写你的命令 ----
accelerate launch --num_processes 8 --num_machines 1 train_hand_motion_prior.py --config configs/hand_motion_prior_flow_matching.yaml