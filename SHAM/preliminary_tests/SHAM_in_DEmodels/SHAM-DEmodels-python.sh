#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J DE
#SBATCH --exclusive

module load openmpi
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=16
module load python/3.7.1

UNITa=('epsi0' 'epsi20' 'epsi50' 'lcdm')
for i in `seq 0 0`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_python/${UNITa[i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir"
  fi
  srun --mpi=pmi2 -n 4 python3 SHAM-zbins-sigma.py ${UNITa[i]} run
done

for i in `seq 0 0`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_python/${UNITa[i]}_Vsmear
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir"
  fi
  srun --mpi=pmi2 -n 4 python3 SHAM-zbins-2param.py ${UNITa[i]} run
done