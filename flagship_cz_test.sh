#!/bin/bash
#
#bash job_init.sh
# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH=~/Documents/Research/python
##$ -o /mnt/lustre/scratch/astro/loveday
# Your job name
##$ -N $@
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
# Job class (test.long = 1 week)
#$ -jc test.long
# specify the queue and number of slots
#$ -q smp.q
##$ -pe openmp 16
#$ -l m_mem_free=32G
# catch kill and suspend signals
#$ -notify
cd /research/astro/gama/loveday/Data/flagship
python <<EOF
import flagship
flagship.cz_test(ranfac=10)
EOF
