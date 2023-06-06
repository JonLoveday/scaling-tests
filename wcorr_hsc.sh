#!/bin/bash
#
#$ -q smp.q
##$ -q serial.q parallel.q smp.q
#$ -pe openmp 16
#$ -l m_mem_free=8G
module load gsl
cd /research/astro/gama/loveday/Data/HSC
python <<EOF
import wcorr
wcorr.w_hsc(nthreads=16)
EOF
