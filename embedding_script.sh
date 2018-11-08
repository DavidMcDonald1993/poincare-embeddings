#!/bin/bash

#SBATCH --job-name=embedding
#SBATCH --output=embedding_%A_%a.out
#SBATCH --error=embedding_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --array 0-959

DATA_DIR=/rds/projects/2018/hesz01/data

ARR=(-dim={5,10,25,50}" "-dset={cora_ml,cora,pubmed,citeseer}" "-seed={0..29}" "-exp={eval_class_pred,eval_lp})

module purge; module load bluebear
module load bear-apps/2018a
module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176

echo "beginning "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embed.py \
		${ARR[${SLURM_ARRAY_TASK_ID}]} \
		-lr 0.3 \
		-epochs 300 \
		-negs 10 \
		-burnin 20 \
		-nproc 0 \
		-distfn poincare \
		-batchsize 10 \
		-eval_each 1 
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
