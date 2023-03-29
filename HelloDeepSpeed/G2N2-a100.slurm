#!/bin/bash
#SBATCH --job-name=g2n2
#SBATCH --gpus=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=00:15:00
#SBATCH --constraint=a100
#SBATCH --reservation=A100

echo "------- JOB Configuration ---------"
echo "scontrol show job ${SLURM_JOBID}"
scontrol show job ${SLURM_JOBID}

echo "------- GPU Configuration ---------"
echo "nvidia-smi -L"
nvidia-smi -L

echo "------- NVLink Configuration ------"
echo "nvidia-smi topo -m"
nvidia-smi topo -m

echo "------- Infiniband Configuration --"
echo "ibv_devinfo"
ibv_devinfo

rm -rf bert_pretrain*

module load dl
module load deepspeed/0.8.3
module list

export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0 
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=4
export MAX_JOBS=${SLURM_CPUS_PER_TASK}

HF=hostfile-${SLURM_JOB_NAME}
rm ${HF}
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

for ((i = 0; i < ${SLURM_NNODES}; i++)); do
   node_i=${nodes_array[$i]}
   echo $node_i slots=${SLURM_GPUS_ON_NODE} >> ${HF}
done

export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${nodes_array[0]} /bin/hostname -I | cut -d " " -f 2)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} python train_bert_ds.py --checkpoint_dir $PWD --num_layers 2 --hidden_dim 512 


