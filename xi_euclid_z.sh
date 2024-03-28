#!/bin/bash
# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH
#$ -o /mnt/lustre/scratch/astro/loveday
# Run job through bash shell
#$ -S /bin/bash
# set mail notification on exit, abort or suspension
#$ -m eas
# who to mail
#$ -M loveday@sussex.ac.uk
# reference from current working directory
#$ -cwd
# Combine error and output files
#$ -j y
# catch kill and suspend signals
#$ -notify
# Job class (test.long = 1 week)
#$ -jc test.long
# specify the queue and number of slots
#
#$ -q smp.q
#$ -pe openmp 16
#$ -l m_mem_free=2G
module load GSL
cd /research/astro/gama/loveday/Data/euclid
python <<EOF
import flagship
flagship.xir_counts()
EOF
