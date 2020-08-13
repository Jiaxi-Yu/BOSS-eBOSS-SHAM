#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=08:00:00
#SBATCH --nodes=16
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell

export OMP_NUM_THREADS=64
srun -n 16 -c 64 python VZsmear-MCMC.py LRG NGC 