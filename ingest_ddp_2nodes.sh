#!/bin/bash
#SBATCH --account=project_462000824
#SBATCH --partition=dev-g
#SBATCH --output=logs/out_ingest_2nodes.txt
#SBATCH --error=logs/error_ingest_2nodes.txt
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=03:00:00

# Create logs directory if it doesn't exist
[ -d logs ] || mkdir logs

# Singularity image
export SIF=/scratch/project_462000824/shanshan/Colqwen.sif

# Load modules
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Hugging Face cache
export HF_HOME="/scratch/project_462000824/${USER}/hf-cache"
mkdir -p $HF_HOME

# Tell RCCL to use Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500



srun singularity exec $SIF bash -c 'python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ingest_ddp.py \
  --images_folder ./images \
  --out_dir ./embeddings_out \
  --batch_size 32 \
  --num_workers 4 \
  '
