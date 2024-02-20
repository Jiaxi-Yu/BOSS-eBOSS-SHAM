#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J CMASS
#SBATCH --exclusive

module load openmpi
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=16

PROJ=('CMASS')
MTSHAMdir=/home/astro/jiayu/Desktop/SHAM/mtsham_percent-cut
#MTSHAMdir1=/home/astro/jiayu/Desktop/SHAM/mtsham_no-cut_debug
MTSHAMdir1=/home/astro/jiayu/Desktop/SHAM/mtsham_no-cut
UNITdir=/hpcstorage/jiayu/UNIT
OBSdir=/home/astro/jiayu/Desktop/SHAM/catalog/BOSS_zbins_mps

UNITa=('0.68620' '0.64210' '0.61420' '0.62800')
zrange=('z0.43z0.51' 'z0.51z0.57' 'z0.57z0.7' 'z0.43z0.7')
params=([0.43872982769911495,0] [0.42161905430552854,0])
zeff=(0.4686 0.5417 0.6399 0.5897)
Ngal=(342000 363000 160000 264000)
#date='0729'
date='0218'
rescale=10
Rrange='15-35'

date
for i in `seq 0 1`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mps_linear_${PROJ}_NGC+SGC_${zrange[$i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir"
  fi

  if [ "$date" == "0218" ]; then
  echo srun --mpi=pmi2 -n 3 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0,0.] --prior-max [0.6,150,0.08] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  else  
  echo srun --mpi=pmi2 -n 3 ${MTSHAMdir1}/MTSHAM --conf ${MTSHAMdir1}/mtsham.conf --prior-min [0,0] --prior-max [0.6,150] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]} 
  # --debug-param ${params[$i]}
  fi
done

# fitting to mocks
#for i in `seq 1 1`; do
  #MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mocks10-mps_linear_${PROJ}_NGC+SGC_${zrange[$i]}
  #MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mean64-mps_linear_${PROJ}_NGC+SGC_${zrange[$i]}
  #if [ ! -d "$MCdir" ]; then
  #mkdir "$MCdir"
  #fi
  #srun --mpi=pmi2 -n 3 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0,0.] --prior-max [0.6,150,0.08] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  #srun --mpi=pmi2 -n 1 ${MTSHAMdir}_debug/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0,0.] --prior-max [0.6,150,0.08] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.mocks10 --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mocks --fit-column [1,2] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC-vceil0.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  
  #srun --mpi=pmi2 -n 1 ${MTSHAMdir1}_debug/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0] --prior-max [1.0,150] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR_mps_linear_${PROJ}_${zrange[$i]}_mocks_quad.${Rrange}.mocks${rescale} --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.${Rrange}.mocks --fit-column [1,2] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]} --fit-index [16,16] --dist-max 35 --dist-bin 35
  #srun --mpi=pmi2 -n 3 ${MTSHAMdir1}/MTSHAM --conf ${MTSHAMdir}/mtsham.conf --prior-min [0,0] --prior-max [0.6,150] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}  --num-sample 64

#done
date