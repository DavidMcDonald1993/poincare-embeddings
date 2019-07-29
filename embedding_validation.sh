#!/bin/bash

for dataset in cora_ml citeseer ppi pubmed mit
do
    for exp in nc_experiment lp_experiment
    do
        for seed in {0..29}
        do
            for dim in 5 10 25 50
            do
                embedding_f=$(printf "emebddings/${dataset}/dim=%02d/seed=%03d/${exp}/embedding.csv" ${dim} ${seed})
                if [ ! -f embedding_f ]
                then 
                    echo no embedding at ${embedding_f}
                fi
            done
        done
    done
done