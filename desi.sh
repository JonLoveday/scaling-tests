#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J desi
#SBATCH --mail-user=j.loveday@sussex.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 24:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#srun -n 64 -c 4 --cpu_bind=cores python <<EOF
module load conda
conda activate jon
python <<EOF
import platform
print(platform.python_version())
import desi
desi.desi_legacy_xcounts()
EOF
