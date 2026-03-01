#!/bin/bash -i
#SBATCH --job-name=test_all_dino_fpn
#SBATCH --partition=gpu     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1   
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --output=%j.out
#SBATCH --error=%j.err

conda activate handtraj
echo "Starting DINO feature extraction for split: 'train', 'test', 'val', dataset: 'dexycb', 'interhand26m', 'arctic', 'ho3d'"

python /public/home/annie/code/handtraj/dataset_preprocessing/process_0227_0206.py