#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J desi
#SBATCH --mail-user=j.loveday@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 64 -c 4 --cpu_bind=cores python <<EOF
import desi
desi.wcounts_S()
EOF
