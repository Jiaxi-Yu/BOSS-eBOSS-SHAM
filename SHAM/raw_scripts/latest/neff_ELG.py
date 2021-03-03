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

def combineNS_ELG(ifile):
  nchunk = 4
  vol = [None] * nchunk
  cnt = [None] * nchunk
  area = []

  with open(ifile) as f:
    for line in f:
      if 'Eff. area' in line:
        area = np.array(line.split()[-nchunk:]).astype(np.float)
        break

  if len(area) != nchunk:
    raise IOError('failed to find eff. area from {}'.format(ifile))

  d = np.loadtxt(ifile, unpack=True)
  if len(d) != nchunk * 2:
    raise IOError('wrong format of file {}'.format(ifile))

  for i in range(nchunk):
    dz = d[i*2][1:] - d[i*2][:-1]
    zint = float('{:.2g}'.format(np.mean(dz)))
    nbin = len(d[i*2])

# Do not know why the first bin is negative, but the effects should be minor
    zl = np.arange(nbin) * zint - zint
    zu = np.arange(nbin) * zint
    z = np.arange(nbin) * zint - zint * 0.5

    vol[i] = np.zeros(nbin)
    cnt[i] = np.zeros(nbin)

    for j in range(nbin):
      vol[i][j] = shell_vol(area[i], zl[j], zu[j])
    cnt[i] = vol[i] * d[i*2+1]

  nz = np.sum(np.array(cnt), axis=0) / np.sum(np.array(vol), axis=0)
  nzN = np.sum(np.array(cnt[:2]), axis=0) / np.sum(np.array(vol[:2]), axis=0)
  nzS = np.sum(np.array(cnt[2:]), axis=0) / np.sum(np.array(vol[2:]), axis=0)

  #return [z, nz]
  return [z, nz,nzN,nzS]

def neff(z, nz, zmin=0, zmax=1):
  sel = (z >= zmin) & (z <= zmax)
  z = z[sel]
  nz = nz[sel]
  chi = np.array([cmv_dist(zi) for zi in z])
  neff2 = simps(nz**2 * chi**2, x=chi) / simps(chi**2, x=chi)
  return np.sqrt(neff2)


z, nz,nzN,nzS = combineNS_ELG('/media/jiaxi/disk/Master/obs/nbar_eBOSS_ELG_v7.dat')
n_eff = neff(z, nz, zmin=zmin, zmax=zmax)
print('NGC+SGC:',n_eff)
n_eff = neff(z, nzN, zmin=zmin, zmax=zmax)
print('NGC:',n_eff)
n_eff = neff(z, nzS, zmin=zmin, zmax=zmax)
print('SGC:',n_eff)

