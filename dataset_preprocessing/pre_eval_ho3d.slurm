#!/bin/bash -i
#SBATCH --job-name=preprocess-ho3d-eval
#SBATCH --partition=gpu     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2   
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --output=%j.out
#SBATCH --error=%j.err

conda activate hamer_feat

python preprocess_ho3d.py