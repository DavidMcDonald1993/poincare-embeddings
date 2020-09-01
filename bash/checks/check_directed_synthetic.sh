#!/bin/bash

for dataset in {00..29}
do
    for exp in recon_experiment lp_experiment
    do
        for seed in 0
        do
            for dim in 5 10 25 50
            do
                embedding_f=$(printf \
                "embeddings/directed/synthetic_scale_free/${dataset}/dim=%02d/seed=%03d/${exp}/embedding.csv" ${dim} ${seed})
                if [ -f ${embedding_f}.gz ]
                then
                    continue
                elif [ -f ${embedding_f} ]
                then 
                    gzip $embedding_f 
                else
                    echo no embedding at ${embedding_f}
                fi
            done
        done
    done
done