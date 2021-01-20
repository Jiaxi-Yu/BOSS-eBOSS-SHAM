#!/usr/bin/env python3
import numpy as np
from scipy.integrate import quad, simps
import sys

c = 299792.458
Om = 0.31
zmin = np.float32(sys.argv[1])
zmax = np.float32(sys.argv[2])


def inv_efunc(z):
  return 1 / np.sqrt(1 - Om + Om * (1+z)**3)

def cmv_dist(z):
  return quad(inv_efunc, 0, z)[0] * c / 100

def shell_vol(area, z1, z2):
  d1 = cmv_dist(z1)
  d2 = cmv_dist(z2)
  vol = np.abs(4 * np.pi / 3. * (d2**3 - d1**3))
  full_area = 4 * np.pi * (180 / np.pi)**2
  return vol * area / full_area

def combineNS(nbar_fmt, sample=['N','S']):
  n = len(sample)
  zs = [None] * n
  vol = [None] * n
  cnt = [None] * n

  for i,sp in enumerate(sample):
    ifile = nbar_fmt.format(sp)
    zs[i], vol[i], cnt[i] = np.loadtxt(ifile, usecols=(0,5,6), unpack=True)

  for i in range(n-1):
    uniq_z = np.unique(zs[i+1]-zs[i])
    if len(uniq_z) != 1:
      raise ValueError('redshift mismatch')

  nz = np.sum(np.array(cnt), axis=0) / np.sum(np.array(vol), axis=0)
  return [zs[0], nz]

def neff(z, nz, zmin=0, zmax=1):
  sel = (z >= zmin) & (z <= zmax)
  z = z[sel]
  nz = nz[sel]
  chi = np.array([cmv_dist(zi) for zi in z])
  neff2 = simps(nz**2 * chi**2, x=chi) / simps(chi**2, x=chi)
  return np.sqrt(neff2)


z, nz = combineNS('/media/jiaxi/disk/Master/nbar_eBOSS_LRG_{}GC_v7_2.dat')
n_eff = neff(z, nz, zmin=zmin, zmax=zmax)

print(n_eff)

