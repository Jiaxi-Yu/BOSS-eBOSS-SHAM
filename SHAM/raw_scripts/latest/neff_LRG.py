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
    nbar = [None] * n
    vol = [None] * n
    cnt = [None] * n
    Veff = [None] * n
    P0 = 10000

    for i,sp in enumerate(sample):
        ifile = nbar_fmt.format(sp)
        zs[i], nbar[i], vol[i], cnt[i] = np.loadtxt(ifile, usecols=(0,3,5,6), unpack=True)
        Veff[i] = (nbar[i]*P0/(nbar[i]*P0+1))**2*vol[i]

    for i in range(n-1):
        uniq_z = np.unique(zs[i+1]-zs[i])
        if len(uniq_z) != 1:
            raise ValueError('redshift mismatch')

    nz = np.sum(np.array(cnt), axis=0) / np.sum(np.array(vol), axis=0)
    return [zs[0], nz,cnt[0]/vol[0],cnt[1]/vol[1],Veff]

def neff(z, nz, sel):
    z = z[sel]
    nz = nz[sel]
    chi = np.array([cmv_dist(zi) for zi in z])
    neff2 = simps(nz**2 * chi**2, x=chi) / simps(chi**2, x=chi)
    return np.sqrt(neff2)

root =  '/home/jiaxi/Desktop/data_archive/'
gal  = sys.argv[1]
if gal == 'eBOSS':
    fileroot = root+'clustering_eBOSS/nbar_eBOSS_LRG_{}GC_v7_2.dat'
    zmins = [0.6,0.65,0.7,0.75]#[0.6,0.6,0.65,0.7,0.8,0.6]
    zmaxs = [0.65,0.7,0.75,0.8]#[0.7,0.8,0.8, 0.9,1.0,1.0]
    z, nz,nzN,nzS,V_eff = combineNS(fileroot)
else:
    fileroot = root+'clustering_BOSS/nbar_DR12v5_'+gal+'_{}_om0p31_Pfkp10000.dat'
    if gal == 'LOWZ':
        zmins = [0.15,0.2, 0.33,0.2]
        zmaxs = [0.2, 0.33,0.43,0.43]
    elif gal == 'CMASS': 
        zmins = [0.43,0.51,0.57,0.43]
        zmaxs = [0.51,0.57,0.70,0.70]
    elif gal == 'CMASSLOWZ':
        zmins = [0.2,0.4,0.5]
        zmaxs = [0.5,0.6,0.75]    
    else:
        print("input should be \'eBOSS\', \'LOWZ\', \'CMASS\' or \'CMASSLOWZ\'")
    z, nz,nzN,nzS,V_eff = combineNS(fileroot,sample = ['North','South'])

for zmin,zmax in zip(zmins,zmaxs):
    sel = (z >= zmin) & (z <= zmax)
    n_eff = neff(z, nz, sel)
    Veff_tot = sum(V_eff[0][sel])+sum(V_eff[1][sel])
    print('z{}z{} NGC+SGC number density: {}e-4, effective volume: {}(Gpc/h)^3,, error ratio:{} \n'.format(zmin,zmax,n_eff*1e4, Veff_tot/1e9, 1+Veff_tot/1e9/10))
    #n_eff = neff(z, nzN, zmin=zmin, zmax=zmax)
    #print('z{}z{} NGC:{}e-4\n'.format(zmin,zmax,n_eff*1e4))
    #n_eff = neff(z, nzS, zmin=zmin, zmax=zmax)
    #print('z{}z{} SGC:{}e-4\n'.format(zmin,zmax,n_eff*1e4))

