#!/bin/bash

#SBATCH --job-name=removeEdges
#SBATCH --output=removeEdges_%A_%a.out
#SBATCH --error=removeEdges_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --array 0-89

# DATA_DIR=/rds/projects/2018/hesz01/data
DATA_DIR="/rds/homes/d/dxm237/data"

module purge; module load bluebear
module load apps/python3/3.5.2

ARR=(--dataset={cora_ml,citeseer,"ppi --only-lcc"}" "--seed={0..29})

echo "beginning "${ARR[${SLURM_ARRAY_TASK_ID}]}
python remove_edges.py ${ARR[${SLURM_ARRAY_TASK_ID}]} --data-directory ${DATA_DIR}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
