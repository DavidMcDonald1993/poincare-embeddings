#!/bin/bash

#SBATCH --job-name=removeEdges
#SBATCH --output=removeEdges_%A_%a.out
#SBATCH --error=removeEdges_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --array 0-239

DATA_DIR=/rds/projects/2018/hesz01/data

module purge; module load bluebear
module load apps/python3/3.5.2

ARR=(--dataset={cora_ml,cora,pubmed,citeseer,AstroPh,CondMat,GrQc,HepPh}" "--seed={0..29})

echo "beginning "${ARR[${SLURM_ARRAY_TASK_ID}]}
python remove_edges.py ${ARR[${SLURM_ARRAY_TASK_ID}]} --data-directory ${DATA_DIR}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
