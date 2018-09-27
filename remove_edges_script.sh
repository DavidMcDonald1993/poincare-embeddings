#!/bin/bash

#SBATCH --job-name=removeEdges
#SBATCH --output=removeEdges_%A_%a.out
#SBATCH --error=removeEdges_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array 0-119

module purge; module load bluebear
module load apps/python3/3.5.2

ARR=(--dataset={cora_ml,cora,pubmed,citeseer}" "--seed={0..29})

python remove_edges.py ${ARR[${SLURM_ARRAY_TASK_ID}]}
