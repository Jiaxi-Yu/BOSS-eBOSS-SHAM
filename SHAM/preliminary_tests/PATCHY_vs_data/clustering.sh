#!/bin/bash

if [ $# != 2 ]; then
  echo "Usage: $0 CAP NUM"
  exit
fi

tstart=`date +%s`

CLSS=('CMASS' 'LOWZ')
#VERS=('DR12v5' 'DR12v5')
# FCFC configuration
CLDIR=/home/astro/jiayu/Desktop/FCFC
CLCONFDIR=${CLDIR}

if [ "$OMP_NUM_THREADS" = "" ]; then
  export OMP_NUM_THREADS=16
fi
ZC2=(0.57)
ZC1=(0.51)
#ZC2=(0.51 0.57 0.7 0.7)
#ZC1=(0.43 0.51 0.57 0.43)
#ZC2=(0.33 0.43 0.43)
#ZC1=(0.2 0.33 0.2)

CAP=$1
NUM=`echo "$2 + 0" | bc`
NUM=`printf %04d $NUM`
if [ "$CAP" != "N" -a "$CAP" != "S" ]; then
  echo "Error: CAP must be N or S."
  exit
fi

for ii in `seq 0 0`; do

if [ "$ii" = 0 -a "${ZC2[0]}" = 1.0 ]; then
  continue
fi

# data files
CLS=${CLSS[$ii]}
#VER=${VERS[$ii]}
VER='DR12'
FINALDIR=/hpcstorage/jiayu/PATCHY/${CLS}
DATA=${FINALDIR}/Patchy-Mocks-${VER}${CLS}-${CAP}-V6C-Portsmouth-mass_${NUM}.dat
RAND=${FINALDIR}/Random-${VER}${CLS}-${CAP}-V6C-x20.dat

for i in `seq 0 0`; do
  zmin=${ZC1[$i]}
  zmax=${ZC2[$i]}
  ODIR=${FINALDIR}/z${zmin}z${zmax}
  if [ ! -d "$ODIR" ]; then
    mkdir "$ODIR"
  fi

  CFDIR=${ODIR}/2PCF
  if [ ! -d "$CFDIR" ]; then
    mkdir "$CFDIR"
  fi

  OUT=PATCHYmock_${CLS}_${CAP}GC_${VER}_z${zmin}z${zmax}_${NUM}
  FILEROOT=${CFDIR}/2PCF_${OUT}
  RRFILE=${CFDIR}/2PCF_PATCHYmock_${CLS}_${CAP}GC_${VER}_z${zmin}z${zmax}.rr

  CDIR="$CLCONFDIR"
  ${CDIR}/FCFC_2PT --conf ${CDIR}/etc/fcfc_2pt_mocks.conf -i "[$DATA,$RAND]" -P "[${FILEROOT}.dd,${FILEROOT}.dr,$RRFILE]" -E "[${FILEROOT}.xi]" -M "[${FILEROOT}.mps]" -s "[\$3>=${zmin}&&\$3<${zmax}&&\$7==1&&\$8==1,\$3>=${zmin}&&\$3<${zmax}&&\$6==1&&\$7==1]"
done

done

tend=`date +%s`
tspend=`echo "( $tend - $tstart ) / 60" | bc`

echo "----- clustering: ${CAP}GC $zmin $zmax ($NUM) DONE in $tspend mins  -----"

