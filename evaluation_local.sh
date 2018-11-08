#!/bin/bash

for seed in {0..29}; do

	for dim in {5,10}; do


		python evaluate_lp.py -dim=${dim} -dset=cora_ml -seed=${seed}

	done


done
