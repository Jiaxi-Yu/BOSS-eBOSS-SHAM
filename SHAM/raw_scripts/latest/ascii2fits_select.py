#!/usr/bin/env python3
import sys
from pandas import read_csv
import numpy as np
from astropy.table import Table

if len(sys.argv) != 3:
  print('Usage: {} INPUT OUTPUT'.format(sys.argv[0]), file=sys.stderr)
  sys.exit(1)


  print('reading source files')
  data = read_csv('hlist_{}.list'.format(sys.argv[1]), header=None, delim_whitespace=True, comment='#', \
      engine='c', usecols=(1,5,17,18,19,22,62), \
      names=('ID','PID','X','Y','Z','VZ','Vpeak'), \
      dtype={'ID':np.int64, 'PID':np.int64,'X':np.float32,'Y':np.float32,'Z':np.float32, 'VZ':np.float32,'Vpeak':np.float32})

  t = Table.from_pandas(data)
  #t.write('UNIT_hlist_{}.fits.gz'.format(sys.argv[1]), format='fits', overwrite=True)
  print('writing the new one')
  t.write('UNIT_hlist_{}.dat'.format(sys.argv[2]), format='ascii.no_header', overwrite=True)