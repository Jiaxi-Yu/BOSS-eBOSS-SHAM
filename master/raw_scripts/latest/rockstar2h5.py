#!/usr/bin/env python3
import numpy as np
from pandas import read_csv
import h5py
import sys

if len(sys.argv) != 3:
    print('Usage: {} INPUT OUTPUT'.format(sys.argv[0]), file=sys.stderr)
    sys.exit(1)

ifile = sys.argv[1]
ofile = sys.argv[2]

data = read_csv(ifile, header=None, delim_whitespace=True, comment='#', \
    engine='c', usecols=(0,1,2,3,4),\
    names=('X','Y','Z','VZ','Vpeak'), \
    dtype={'X':np.float32, 'Y':np.float32,'Z':np.float32,'VZ':np.float32,'Vpeak':np.float32})
#usecols=(1,5,10,16,17,18,19,20,21,22), \
#names=('ID','PID','Mvir','Vmax','X','Y','Z','VX','VY','VZ'), \
#dtype={'ID':np.int64, 'PID':np.int64, 'Mvir':np.float32,'Vpeak':np.float32, 'X':np.float32, 'Y':np.float32,'Z':np.float32,'VX':np.float32, 'VY':np.float32, 'VZ':np.float32})
with h5py.File(ofile, 'w') as ff:
    ff.create_group('halo')
    ff.create_dataset('halo/X', data=data['X'].values)
    ff.create_dataset('halo/Y', data=data['Y'].values)
    ff.create_dataset('halo/Z', data=data['Z'].values)
    ff.create_dataset('halo/VZ', data=data['VZ'].values)
    ff.create_dataset('halo/Vpeak', data=data['Vpeak'].values)
