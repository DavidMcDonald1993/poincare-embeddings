#!/bin/bash

#SBATCH --job-name=NK
#SBATCH --output=NK_%A_%a.out
#SBATCH --error=NK_%A_%a.err
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --array 0-1499

dims=(2 5 10 25 50)
datasets=(cora_ml citeseer ppi mit pubmed)
seeds=({0..29})
exps=({nc,lp}_experiment)

num_dims=${#dims[@]}
num_datasets=${#datasets[@]}
num_seeds=${#seeds[@]}
num_exps=${#exps[@]}

dim_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds * num_datasets) % num_dims))
dataset_id=$((SLURM_ARRAY_TASK_ID / (num_exps * num_seeds) % num_datasets))
seed_id=$((SLURM_ARRAY_TASK_ID / num_exps % num_seeds))
exp_id=$((SLURM_ARRAY_TASK_ID % num_exps))

dim=${dims[$dim_id]}
dataset=${datasets[$dataset_id]}
seed=${seeds[$seed_id]}
exp=${exps[$exp_id]}

echo $dim $dataset $seed $exp

if [ $exp == nc_experiment ]
then
    edgelist_filename=../heat/datasets/${dataset}/edgelist.tsv
else 
    edgelist_filename=$(printf ../heat/edgelists/${dataset}/seed=%03d/training_edges/edgelist.tsv ${seed})
fi

embedding_directory=$(printf embeddings/undirected/${dataset}/dim=%02d/seed=%03d/${exp} ${dim} ${seed})
embedding_filename=${embedding_directory}/embedding.csv.gz

if [ ! -f ${embedding_filename} ]
then

    module purge; module load bluebear
    module load bear-apps/2018a
    module load PyTorch/0.4.0-foss-2018a-Python-3.6.3-CUDA-9.0.176

    echo making embedding at $embedding_filename

    python embed.py \
        -dset ${dataset} \
        -dim ${dim} \
    	-edgelist ${edgelist_filename} \
        -embedding_dir ${embedding_directory} \
    	-lr 1.0 \
    	-epochs 1500 \
    	-negs 10 \
    	-burnin 20 \
    	-nproc 0 \
    	-distfn poincare \
    	-batchsize 1024 \
    	-eval_each 1500 
    # echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}

else    
    echo $embedding_filename exists 
fi
