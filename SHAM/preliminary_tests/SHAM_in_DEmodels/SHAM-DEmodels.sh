#!/bin/bash

#SBATCH -p p4
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH -J BOSS
#SBATCH --exclusive

module load openmpi
export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=16

PROJ=('CMASSLOWZTOT')
tail='DEmodel'
MTSHAMdir=/home/astro/jiayu/Desktop/SHAM/mtsham_percent-cut
UNITdir=/hpcstorage/zhaoc/Galileon/Cubic_Galileon_Halo
OBSdir=/home/astro/jiayu/Desktop/SHAM/catalog/BOSS_zbins_mps

UNITa=('epsi0' 'epsi20' 'epsi50' 'lcdm')
zrange=('z0.2z0.75')
zeff=(0.5609)
Ngal=(26000)

date
for i in `seq 0 3`; do
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_0218/mps_linear_${tail}_NGC+SGC_${zrange}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir"
  fi
  srun --mpi=pmi2 -n 4 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham_DE.conf --prior-min [0,60,0.] --prior-max [1,180,0.15] -i ${UNITdir}/${UNITa[$i]}_snapshot_009.z0.500.AHF_halos --cov-matrix ${OBSdir}/covR-${PROJ}_NGC+SGC-s5_25-${zrange[$i]}-quad.dat --fit-ref ${OBSdir}/OBS_${PROJ}_NGC+SGC_DR12v5_${zrange[$i]}.mps --fit-column [4,5] --best-fit ${MCdir}/best-fit_${tail}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
done
date
