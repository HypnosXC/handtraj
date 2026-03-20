#!/bin/bash

#SBATCH --job-name=handgen_32gpu
#SBATCH --partition=gpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1       # 1 task per node; accelerate handles GPU processes
#SBATCH --gres=gpu:8              # 8 GPUs per node
#SBATCH --cpus-per-task=64        # all CPUs on the node available to accelerate's 8 workers
#SBATCH --time=96:00:00

ulimit -l unlimited
ulimit -s unlimited

# -------------------------------
# 1. Environment
# -------------------------------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# -------------------------------
# 2. Distributed training env vars
# -------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node List:    $SLURM_JOB_NODELIST"
echo "Master Addr:  $MASTER_ADDR"
echo "Master Port:  $MASTER_PORT"
echo "Num Nodes:    $SLURM_NNODES"
echo "GPUs/node:    $SLURM_GPUS_ON_NODE"
echo "========================================="

# -------------------------------
# 3. NCCL
# -------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=5

# -------------------------------
# 4. Activate env
# -------------------------------
srun bash -c 'echo "hostname=$(hostname) SLURM_NODEID=$SLURM_NODEID MASTER_ADDR='"$MASTER_ADDR"'"'

# -------------------------------
# 5. Launch training (2 nodes x 8 GPUs = 16 processes)
# -------------------------------
srun bash -c 'accelerate launch \
    --num_processes 16 \
    --num_machines 2 \
    --machine_rank $SLURM_NODEID \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    --multi_gpu \
    train_hand_motion_prior.py \
        --config configs/hand_motion_prior_flow_matching.yaml'