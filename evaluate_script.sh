#!/bin/bash

#SBATCH --job-name=removeEdges
#SBATCH --output=removeEdges_%A_%a.out
#SBATCH --error=removeEdges_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --array 0-1199

DATA_DIR=/rds/projects/2018/hesz01/data

module purge; module load bluebear
module load apps/python3/3.5.2

ARR=(-dim={5,10,20,50,100}" "-dset={cora_ml,cora,pubmed,citeseer,AstroPh,CondMat,GrQc,HepPh}" "-seed={0..29})

python evaluate_lp.py ${ARR[${SLURM_ARRAY_TASK_ID}]} --data-directory ${DATA_DIR}

