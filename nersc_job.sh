#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=j.loveday@sussex.ac.uk

#module load conda
#conda activate jon
python <<EOF
import legacy
legacy.select()
EOF

