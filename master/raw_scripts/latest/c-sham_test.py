#!/usr/bin/env python3
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import time

multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home      = '/home/astro/jiayu/Desktop/HAM-MCMC/'

z = 0.7018
a_t = '0.58760'
ver = 'v7_2'
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)
bins  = np.arange(rmin,rmax+1,1)
nbins = len(bins)-1
binmin = rmin
binmax = rmax
# generate mu bins
s = (bins[1:]+bins[:-1])/2
mubins = np.linspace(0,1,121)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

halofile = home+'catalog/UNIT4LRG-cut200.dat'
print('reading UNIT catalog')
a=time.time()
LRGscat = Table.read(halofile,format='ascii.no_header')
SHAMnum = 62600#len(LRGscat)
print('SHAMnum = {}'.format(SHAMnum))
b=time.time()
print('reading costs {}s'.format(b-a))
# make sure len(data) is even
boxsize=1000
a = time.time()
LRGscat = LRGscat[argpartition(-LRGscat['col5'],SHAMnum)[:(SHAMnum)]]
z_redshift  = LRGscat['col3']+(LRGscat['col4']*(1+z)/H)
z_redshift %=boxsize
# calculate the 2pcf of the SHAM galaxies
# count the galaxy pairs and normalise them
DD_counts = DDsmu(1,16,bins,1, nmu,LRGscat['col1'],LRGscat['col2'],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
# calculate the 2pcf and the multipoles
mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
Table([np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu]).write(home+'LRG_NGC-redshift_space_cutpart-python.dat',format = 'ascii.no_header',delimiter='\t',overwrite=True)
b=time.time()
print('calculate and save 2pcf cost {}s'.format(b-a))
DD_counts1 = DDsmu(1,16,bins,1,nmu,LRGscat['col1'],LRGscat['col2'],LRGscat['col3'],periodic=True, verbose=True,boxsize=boxsize)
# calculate the 2pcf and the multipoles
mono1 = (DD_counts1['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
quad1 = mono1 * 2.5 * (3 * mu**2 - 1)
hexa1 = mono1 * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)

#Table([np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu]).write(home+'LRG_NGC-redshift_space_sigma{}_sigmaV{}_Vceil{}_cutall-python.dat'.format(sig,sigV,Vceil),format = 'ascii.no_header',delimiter='\t',overwrite=True)
Table([np.sum(mono1,axis=-1)/nmu,np.sum(quad1,axis=-1)/nmu,np.sum(hexa1,axis=-1)/nmu]).write(home+'LRG_NGC-real_space_cutpart-python.dat',format = 'ascii.no_header',delimiter='\t',overwrite=True)
