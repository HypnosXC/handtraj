#!/bin/bash

#SBATCH --job-name=handgen_multigpu
#SBATCH --partition=gpu
#SBATCH --output=%j.out
#SBATCH --error=%j.err
# 总进程数 = 节点数 * 每节点进程数
#SBATCH --nodes=4                 # 【重要】修改这里：指定使用的机器数量 (例如 2 台)
#SBATCH --ntasks-per-node=8       # 每个节点启动 8 个进程 (对应 8 张卡)
#SBATCH --gres=gpu:8              # 【重要】每个节点申请 8 张 GPU
#SBATCH --cpus-per-task=8         # 每个进程绑定的 CPU 核心数 (配合 OMP_NUM_THREADS)
#SBATCH --time=24:00:00           # 任务最大运行时间

# 解除资源限制
ulimit -l unlimited
ulimit -s unlimited

# -------------------------------
# 1. 基础环境配置
# -------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# -------------------------------
# 2. 分布式训练环境变量 (PyTorch DDP 标准)
# -------------------------------
# SLURM 自动生成的变量
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500  # 可根据实际情况修改，确保端口未被占用
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((SLURM_PROCID / SLURM_NTASKS_PER_NODE))

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Global Rank: $RANK"
echo "Local Rank: $LOCAL_RANK"
echo "Node Rank: $NODE_RANK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "========================================="

# -------------------------------
# 3. NCCL 通信库优化 (可选，针对 NVIDIA GPU)
# -------------------------------
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 如果使用 InfiniBand 网络设为 0，纯以太网可设为 1
export NCCL_SOCKET_IFNAME=eth0    # 根据实际网卡名称修改 (如 ib0, eth0, bond0)
export NCCL_NET_GDR_LEVEL=5       # 启用 GPUDirect RDMA (如果硬件支持)

# ---- 在这里加载环境 ----
source activate handtraj

# ---- 在这里写你的命令 ----
accelerate launch --num_processes 32 \
                  --num_machines 4 \
                  --machine_rank $SLURM_NODEID \
                  --main_process_ip $MASTER_ADDR \
                  --main_process_port $MASTER_PORT \
                  train_hand_motion_prior.py \
                    --config.experiment-name image_visual_sublen_128 \
                    --config.batch_size 1024 \
                    --config.subseq_len 128 \