#!/bin/bash
#SBATCH --job-name=g4n1
#SBATCH --gpus=4
#SBATCH --gpus-per-node=2
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
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


module load dl
module load deepspeed/0.8.3
module list

export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0 
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=4
export NCCL_TOPO_DUMP_FILE=./nccl_dump.log.${SLURM_JOB_NAME}
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
rm hostfile
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

for ((i = 0; i < ${SLURM_NNODES}; i++)); do
   node_i=${nodes_array[$i]}
   echo $node_i slots=${SLURM_GPUS_ON_NODE} >> hostfile
done

export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${nodes_array[0]} /bin/hostname -I | cut -d " " -f 2)

##### run-benchmark-model.sh Starts here #####
./run-big-model.sh
