#!/bin/bash
#
#$ -q smp.q
##$ -q serial.q parallel.q smp.q
#$ -pe openmp 16
#$ -l h_vmem=16G
module load gsl
cd /research/astro/gama/loveday/Data/euclid
python <<EOF
import flagship
flagship.xi_M_z(randist='shuffle', nthreads=16)
EOF
