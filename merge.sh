#!/bin/bash
#SBATCH --account=project_462000824
#SBATCH --partition=small
#SBATCH --output=logs/out_merge.txt
#SBATCH --error=logs/error_merge.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00

[ -d logs ] || mkdir logs

export SIF=/scratch/project_462000824/shanshan/Colqwen.sif

# Load modules
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

srun singularity exec $SIF bash -c "python merge.py --in_dir './embeddings_out/' --out_dir 'merged_files'"
