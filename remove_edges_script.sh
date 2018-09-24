#!/bin/bash

#SBATCH --job-name=removeEdges
#SBATCH --output=removeEdges.out
#SBATCH --error=removeEdges.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

module purge; module load bluebear
module load apps/python3/3.5.2


python remove_edges.py
