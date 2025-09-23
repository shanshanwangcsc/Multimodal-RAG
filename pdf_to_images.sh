#!/bin/bash
#SBATCH --account=project_462000824
#SBATCH --partition=standard
#SBATCH --output=logs/out_pdf_to_image.txt
#SBATCH --error=logs/error_pdf_to_image.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=6:00:00

[ -d logs ] || mkdir logs

export SIF=/scratch/project_462000824/shanshan/Colqwen.sif
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings


srun singularity exec $SIF bash -c "python pdf_to_images.py"
