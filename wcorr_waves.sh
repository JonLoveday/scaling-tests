#!/bin/bash
#
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
# Job class (test = 8 hours, test.long = 1 week)
#$ -jc test.long
#$ -q smp.q
#$ -pe openmp 16
#$ -l m_mem_free=2G
conda deactivate
conda activate jon
echo $PATH
cd /research/astro/gama/loveday/Data/4MOST/WAVES/target_cats_2022_nov_25
python <<EOF
import waves
waves.wcounts_S()
EOF
