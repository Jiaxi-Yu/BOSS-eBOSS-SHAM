from astropy.table import Table
import numpy as np
from glob import glob
import sys

gal =sys.argv[1] 
if gal=='LRG':
    route = '/global/homes/z/zhaoc/cscratch/EZmock/LRG/EZmock_eBOSS_LRG_v7/EZmock_eBOSS_LRG_{}_v7_{}.dat'
else:
    route = '/global/homes/z/zhaoc/cscratch/EZmock/ELG/nosel_v5/fkp_cap/EZmock_eBOSS_ELG_{}_v7_{}.dat'

files = glob(route.format('NGC','*'))
Ngal = np.zeros((len(files),2))
for i,GC in enumerate(['NGC','SGC']):
    for N in range(len(files)):
        path = route.format(GC,str(N+1).zfill(4))
        cata = np.loadtxt(path)
        Ngal[N,i] = np.sum(cata[:,3])
        if N%100==0:
            print('{}% completed'.format(N/10))
            print(Ngal[N,i])

np.savetxt('EZmocks_{}_Ngal.dat'.format(gal),Ngal)


