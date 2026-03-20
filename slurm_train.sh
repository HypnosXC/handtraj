#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
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
accelerate launch --num_processes 8 --num_machines 1 train_hand_motion_prior.py --config.experiment-name image_visual_sublen_128 --config.batch_size 256 --config.subseq_len 128