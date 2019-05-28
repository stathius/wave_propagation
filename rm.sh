#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=MSC
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-10:00:00
rm -rf /disk/scratch/s1680171/wave_propagation