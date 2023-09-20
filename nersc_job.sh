#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu

srun python <<EOF
import desi
desi.legacy_wcounts()
EOF

