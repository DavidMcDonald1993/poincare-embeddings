#!/bin/bash

#SBATCH --job-name=NK
#SBATCH --output=NK_%A_%a.out
#SBATCH --error=NK_%A_%a.err
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --array 0-1199

ARR=(-dim={50,5,10,25}" "-dset={pubmed,cora_ml,citeseer,ppi,mit}" "-seed={0..29}" "-exp={lp_experiment,nc_experiment})

module purge; module load bluebear
module load bear-apps/2018a
module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176

echo "beginning "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embed.py \
	${ARR[${SLURM_ARRAY_TASK_ID}]} \
	-lr 1.0 \
	-epochs 1500 \
	-negs 10 \
	-burnin 20 \
	-nproc 0 \
	-distfn poincare \
	-batchsize 1024 \
	-eval_each 1500 
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
