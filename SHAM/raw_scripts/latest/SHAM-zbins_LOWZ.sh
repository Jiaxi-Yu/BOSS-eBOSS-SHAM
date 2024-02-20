#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J LOWZ
#SBATCH --exclusive

module load openmpi
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=16

PROJ=('LOWZ')
MTSHAMdir=/home/astro/jiayu/Desktop/SHAM/mtsham_percent-cut
MTSHAMdir1=/home/astro/jiayu/Desktop/SHAM/mtsham_no-cut
UNITdir=/hpcstorage/jiayu/UNIT
OBSdir=/home/astro/jiayu/Desktop/SHAM/catalog/BOSS_zbins_mps

UNITa=('0.78370' '0.71730' '0.74980')
zrange=('z0.2z0.33' 'z0.33z0.43' 'z0.2z0.43')
zeff=(0.2754 0.3865 0.3441)
Ngal=(337000 258000 295000)
GC=('NGC+SGC')
#(NGC SGC)
Rrange='15-35'
date='0218'

date
for i in `seq 2 2`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/Rrange10-25_mps_linear_${PROJ}_NGC+SGC_${zrange[$i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir";
  fi
  srun --mpi=pmi2 -n 1 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham_10-25.conf --prior-min [0,0,0.] --prior-max [0.5,150,0.02] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_${GC}-s10_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_${GC}_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_${GC}.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
done
"""
for i in `seq 0 0`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mps_linear_${PROJ}_NGC+SGC_${zrange[$i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir";
  fi

  if [ "$date" == "0218" ]; then
  echo srun --mpi=pmi2 -n 1 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0,0.] --prior-max [0.5,150,0.02] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_${GC}-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_${GC}_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_${GC}.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  else
  echo srun --mpi=pmi2 -n 4 ${MTSHAMdir1}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0] --prior-max [0.5,150] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  fi
done
"""

# fitting to mocks
"""
for i in `seq 0 2`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mocks10-mps_linear_${PROJ}_${GC}_${zrange[$i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir";
  fi

  if ["$date" -eq '0218']; then
  srun --mpi=pmi2 -n 1 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0,0.] --prior-max [0.5,150,0.02] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.mocks10 --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mocks --fit-column [1,2] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}  --fit-index [16,16] --dist-max 35 --dist-bin 35
  else
  srun --mpi=pmi2 -n 1 ${MTSHAMdir1}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0] --prior-max [1.0,150] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR_mps_linear_${PROJ}_${zrange[$i]}_mocks_quad.${Rrange}.mocks10 --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.${Rrange}.mocks --fit-column [1,2] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}  --fit-index [16,16] --dist-max 35 --dist-bin 35
  fi
done
"""
date