#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from astropy.table import Table

if len(sys.argv) != 3:
  print('Usage: {} INPUT OUTPUT'.format(sys.argv[0]), file=sys.stderr)
  sys.exit(1)

d = pd.read_csv(sys.argv[1], delim_whitespace=True, header=None, \
    names=['X','Y','Z','vx','vy','vz','vpeak'], \
    dtype=np.float32, \
    engine='c')

t = Table.from_pandas(d)
t.write(sys.argv[2], format='fits', overwrite=True)
