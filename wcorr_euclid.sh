#!/bin/bash
#
#$ -q smp.q
##$ -q serial.q parallel.q smp.q
#$ -pe openmp 16
#$ -l m_mem_free=16G
#$ -jc test.long
module load GSL
cd /research/astro/gama/loveday/Data/euclid
python <<EOF
import flagship
flagship.w_mag(nthreads=16)
EOF
