#!/usr/bin/env python3
import numpy as np
from scipy.integrate import quad, simps
import sys

c = 299792.458
Om = 0.31

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
    return [zs[0], nz,cnt[0]/vol[0],cnt[1]/vol[1]]

def neff(z, nz, zmin=0, zmax=1):
    sel = (z >= zmin) & (z <= zmax)
    z = z[sel]
    nz = nz[sel]
    chi = np.array([cmv_dist(zi) for zi in z])
    neff2 = simps(nz**2 * chi**2, x=chi) / simps(chi**2, x=chi)
    return np.sqrt(neff2)

root =  '/global/cscratch1/sd/jiaxi/SHAM/catalog/BOSS_data/'
if sys.argv[1] == 'LOWZ':
    fileroot = root+'nbar_DR12v5_LOWZ_{}_om0p31_Pfkp10000.dat'
    zmins = [0.15,0.2, 0.33,0.2]
    zmaxs = [0.2, 0.33,0.43,0.43]
    z, nz,nzN,nzS = combineNS(fileroot,sample = ['North','South'])

elif sys.argv[1] == 'CMASS': 
    fileroot = root+'nbar_DR12v5_CMASS_{}_om0p31_Pfkp10000.dat'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.70,0.70]
    z, nz,nzN,nzS = combineNS(fileroot,sample = ['North','South'])
    
elif sys.argv[1] == 'CMASSLOWZ':
    fileroot = root+'nbar_DR12v5_CMASS_{}_om0p31_Pfkp10000.dat'
    zmins = [0.2]
    zmaxs = [0.75]
    z, nz,nzN,nzS = combineNS(fileroot,sample = ['North','South'])
    
elif sys.argv[1] == 'eBOSS':
    fileroot = '/media/jiaxi/disk/Master/obs/nbar_eBOSS_LRG_{}GC_v7_2.dat'
    zmins = [0.6,0.6,0.65,0.7,0.8]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0]
    z, nz,nzN,nzS = combineNS(fileroot)
else:
    print("input should be \'eBOSS\', \'LOWZ\', \'CMASS\' or \'CMASSLOWZ\'")

for zmin,zmax in zip(zmins,zmaxs):
    n_eff = neff(z, nz, zmin=zmin, zmax=zmax)
    print('z{}z{} NGC+SGC:{}e-4\n'.format(zmin,zmax,n_eff*1e4))
    #n_eff = neff(z, nzN, zmin=zmin, zmax=zmax)
    #print('z{}z{} NGC:{}e-4\n'.format(zmin,zmax,n_eff*1e4))
    #n_eff = neff(z, nzS, zmin=zmin, zmax=zmax)
    #print('z{}z{} SGC:{}e-4\n'.format(zmin,zmax,n_eff*1e4))

