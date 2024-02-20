#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --nodes=16
#SBATCH -J LRG
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell

# data  & UNIT simulations
gal=(LOWZ LOWZ CMASS CMASS CMASS)
zmin=(0.2 0.33 0.43 0.51 0.57)
zmax=(0.33 0.43 0.51 0.57 0.7)

export OMP_NUM_THREADS=32
for g in `seq 0 4`; do
    srun --cpu-bind=none shifter --image=docker:jiaxiyu/sham-env:jiaxi --env=LD_LIBRARY_PATH=/home/MultiNest/lib:/home/fftw-3.3.10/lib  --env=PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin python3 SHAM-SDSS-plot.py ${gal[$g]} ${zmin[$g]} ${zmax[$g]} 10
done