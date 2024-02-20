#!/bin/bash

#SBATCH -p p5
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=32
#SBATCH -J LRGtest
#SBATCH --exclusive

export NUM_THREADSPROCESSES=${SLURM_CPUS_PER_TASK}
module load openmpi
export OMP_NUM_THREADS=32

PROJ=('LRG')
MTSHAMdir=/home/astro/jiayu/Desktop/SHAM/mtsham_percent-cut
MTSHAMdir1=/home/astro/jiayu/Desktop/SHAM/mtsham_no-cut
UNITdir=/hpcstorage/jiayu/UNIT
OBSdir=/home/astro/jiayu/Desktop/SHAM/catalog/nersc_zbins_wp_mps_LRG

zrange=('z0.6z0.8' 'z0.6z0.7' 'z0.65z0.8' 'z0.7z0.9' 'z0.8z1.0')
Zrange=('0.6-0.8' '0.6-0.7' '0.65-0.8' '0.7-0.9' '0.8-1.0')

Ngal=(88600 93900 88000 64700 30100)
zeff=(0.7051 0.6518 0.7273 0.7968 0.8777)
UNITa=('0.58760' '0.60080' '0.57470' '0.54980' '0.52600')

Rrange='12-40'
rescale='10'
date='0218'
srun --mpi=pmi2 -n 1 python3 SHAM-zbins-plot-percent.py LRG NGC+SGC log 0.6 0.8 1 livep-lolerance_ 0218

date
for i in `seq 0 0`; do 
  MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/norm-mps_log_${PROJ}_NGC+SGC_${zrange[$i]}
  if [ ! -d "$MCdir" ]; then
  mkdir "$MCdir"
  fi
  srun --mpi=pmi2 -n 2 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham_log.conf --prior-min [0,0,0.] --prior-max [1.,150,0.2] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR_mps_log_${PROJ}_${zrange[$i]}_mocks_quad.dat --fit-ref ${OBSdir}/mps_log_${PROJ}_NGC+SGC_eBOSS_v7_2_zs_${Zrange[$i]}.dat --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]} 
  #--tol 0.1 --nlive 1000
done


#date='0726'
#for i in `seq 0 4`; do
  #MCdir=/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_${date}/mocks${rescale}_${Rrange}-mps_log_${PROJ}_NGC+SGC_${zrange[$i]}
  #if [ ! -d "$MCdir" ]; then
  #mkdir "$MCdir"
  #fi
  #if [ "$date" == "0726" ]; then
  #srun --mpi=pmi2 -n 4 ${MTSHAMdir}/MTSHAM --conf ${MTSHAMdir}/mtsham_log.conf --prior-min [0,0,0.] --prior-max [1.0,150,0.2] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR_mps_log_${PROJ}_${zrange[$i]}_mocks_quad.${Rrange}.mocks${rescale} --fit-ref ${OBSdir}/mps_log_${PROJ}_NGC+SGC_eBOSS_v7_2_zs_${Zrange[$i]}.${Rrange}.mocks --fit-column [1,2] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]} --fit-bin [10,10] --dist-file /home/astro/jiayu/Desktop/SHAM/mtsham_percent-cut/binfile_SHAM_${Rrange}_fine.dat #--num-sample 64
  #fi
  #if [ "$date" == "0729" ]; then
  #echo srun --mpi=pmi2 -n 4 ${MTSHAMdir1}/MTSHAM --conf ${MTSHAMdir}/mtsham_log.conf --prior-min [0,0] --prior-max [10.,180] -i ${UNITdir}/UNIT_hlist_${UNITa[$i]}.dat --cov-matrix ${OBSdir}/covR_mps_log_${PROJ}_${zrange[$i]}_mocks_quad.dat --fit-ref ${OBSdir}/mps_log_${PROJ}_NGC+SGC_eBOSS_v7_2_zs_${Zrange[$i]}.dat --fit-column [4,5] --best-fit ${MCdir}/best-fit_${PROJ}_NGC+SGC.dat --output ${MCdir}/multinest_ --redshift ${zeff[$i]} -n ${Ngal[$i]}
  #fi
#done

date

# the upper choice for the modified version
# the lower choice for the original version