#!/bin/bash

#SBATCH --job-name=embedding
#SBATCH --output=embedding_%A_%a.out
#SBATCH --error=embedding_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --array 0-0

DATA_DIR=/rds/projects/2018/hesz01/data

ARR=(-dim={10,20}" "-dset=cora_ml" "-seed={0..100})

module purge; module load bluebear
module load Python/3.6.3-iomkl-2018a
module load bear-apps/2018a
module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176

echo "beggining "${ARR[${SLURM_ARRAY_TASK_ID}]}
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
		-eval_each 1 
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
