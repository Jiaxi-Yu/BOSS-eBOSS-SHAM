#!/usr/bin/env python3
import sys
from pandas import read_csv
import numpy as np
from astropy.table import Table

if len(sys.argv) != 3:
  print('Usage: {} INPUT OUTPUT'.format(sys.argv[0]), file=sys.stderr)
  sys.exit(1)

data = read_csv(sys.argv[1], header=None, delim_whitespace=True, comment='#', \
    engine='c', usecols=(1,5,10,11,12,16,17,18,19,20,21,22), \
    names=('ID','PID','Mvir','Rvir','Rs','Vpeak','X','Y','Z','VX','VY','VZ'), \
    dtype={'ID':np.int64, 'PID':np.int64, 'Mvir':np.float32, 'Rvir':np.float32, \
        'Rs':np.float32,'Vpeak':np.float32, 'X':np.float32, 'Y':np.float32,\
           'Z':np.float32,'VX':np.float32, 'VY':np.float32, 'VZ':np.float32})

t = Table.from_pandas(data)
t.write(sys.argv[2], format='fits', overwrite=True)