#!/usr/bin/env python3
import sys
from pandas import read_csv
import numpy as np
from astropy.table import Table

if len(sys.argv) != 2:
  print('Usage: {} UNITa(t)'.format(sys.argv[0]), file=sys.stderr)
  sys.exit(1)


print('reading source files')
data = read_csv('hlist_{}.list'.format(sys.argv[1]), header=None, delim_whitespace=True, comment='#', \
    engine='c', usecols=(17,18,19,22,62), \
    names=('X','Y','Z','VZ','Vpeak'), \
    dtype={'X':np.float32,'Y':np.float32,'Z':np.float32, 'VZ':np.float32,'Vpeak':np.float32})
# ID and PID are colume 1 and 5
t = Table.from_pandas(data)
print('writing the new one')
t.write('UNIT_hlist_{}.dat'.format(sys.argv[1]), format='ascii.no_header', overwrite=True)