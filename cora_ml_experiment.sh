#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=0
fi

echo "Using $NTHREADS threads"

python embed.py \
       -dim 3 \
       -lr 0.3 \
       -epochs 500 \
       -negs 10 \
       -burnin 10 \
       -nproc "${NTHREADS}" \
       -distfn poincare \
       -dset cora_ml.edg \
       -fout cora_ml.out \
       -batchsize 512 \
       -eval_each 1 \
       -ndproc 5 \
