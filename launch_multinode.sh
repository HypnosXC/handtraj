#!/bin/bash
# Launch 32-GPU training across 4 nodes without slurm.
# Usage: bash launch_multinode.sh

MASTER_ADDR="10.10.13.23"
MASTER_PORT=29500
NUM_MACHINES=4
NUM_PROCESSES=32
CONFIG="configs/hand_motion_prior_flow_matching.yaml"
CONDA_PATH="/public/home/xuchen/anaconda3"

SERVERS=(server23 server24 server25 server27)

ACTIVATE="export PATH=${CONDA_PATH}/bin:\$PATH && eval \"\$(conda shell.bash hook 2>/dev/null)\" && conda activate handtraj"

for i in "${!SERVERS[@]}"; do
    NODE="${SERVERS[$i]}"
    RANK=$i
    echo "Launching on ${NODE} (machine_rank=${RANK})..."

    ssh ${NODE} "cd handtraj && git pull origin main 2>/dev/null; \
        ${ACTIVATE} && \
        ulimit -n 65536 2>/dev/null; \
        export TORCH_NCCL_BLOCKING_WAIT=1; \
        export NCCL_SOCKET_IFNAME=ens22f0np0; \
        export GLOO_SOCKET_IFNAME=ens22f0np0; \
        nohup accelerate launch \
            --num_processes ${NUM_PROCESSES} \
            --num_machines ${NUM_MACHINES} \
            --machine_rank ${RANK} \
            --main_process_ip ${MASTER_ADDR} \
            --main_process_port ${MASTER_PORT} \
            --multi_gpu \
            train_hand_motion_prior.py \
                --config ${CONFIG} \
            > train_run_node${RANK}.log 2>&1 &
        echo 'PID='\$!" &
done

wait
echo "All nodes launched."
