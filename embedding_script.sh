#!/bin/bash

#SBATCH --job-name=embedding
#SBATCH --output=removeEdges_%A_%a.out
#SBATCH --error=removeEdges_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array 1-1

DATA_DIR=/rds/projects/2018/hesz01/data

ARR=(-dim={10,20}" "-dset=cora_ml" "-seed={0..100})

module purge; module load bluebear
# module load apps/python3/3.5.2
module load bear-apps/2018a
module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176


python embed.py \
		${ARR[${SLURM_ARRAY_TASK_ID}]} \
		-lr 0.3 \
		-epochs 300 \
		-negs 10 \
		-burnin 20 \
		-nproc 0 \
		-distfn poincare \
		-fout mammals.pth \
		-batchsize 10 \
		-eval_each 1 \
