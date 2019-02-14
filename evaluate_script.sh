#!/bin/bash

#SBATCH --job-name=evaluation
#SBATCH --output=evaluation_%A_%a.out
#SBATCH --error=evaluation_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --array 0-719

# DATA_DIR=/rds/projects/2018/hesz01/data
DATA_DIR="/rds/homes/d/dxm237/data"

module purge; module load bluebear
# module load bear-apps/2018a
# module load Python/3.6.3-iomkl-2018a
module load apps/python3/3.5.2
# module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176


ARR=(-dim={5,10,25,50}" "-dset={cora_ml,citeseer,"ppi --only-lcc"}" "-seed={0..29}" "-exp={eval_lp,eval_class_pred})

echo "beginning "${ARR[${SLURM_ARRAY_TASK_ID}]}
python evaluate.py ${ARR[${SLURM_ARRAY_TASK_ID}]} --data-directory ${DATA_DIR}
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}

