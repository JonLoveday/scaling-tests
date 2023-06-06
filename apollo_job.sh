#!/bin/bash
#
# This script simply submits the job script specified by command line argument
#
# import required environment variables such as PYTHONPATH
#$ -v PYTHONPATH=~/Documents/Research/python
#$ -o /mnt/lustre/scratch/astro/loveday
# Your job name
#$ -N $@
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
#$ -pe openmp 150
#$ -l m_mem_free=8G
# catch kill and suspend signals
#$ -notify

# Execute script as specified by argument
bash $@
