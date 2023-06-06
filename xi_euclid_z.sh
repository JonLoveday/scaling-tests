#!/bin/bash
# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH=~/Documents/Research/python
#$ -o /mnt/lustre/scratch/astro/loveday
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
# catch kill and suspend signals
#$ -notify
#
#$ -q smp.q
##$ -q serial.q parallel.q smp.q
#$ -pe openmp 32
#$ -l h_vmem=64G
module load GSL
cd /research/astro/gama/loveday/Data/euclid
python <<EOF
import flagship
flagship.xir_counts()
EOF
