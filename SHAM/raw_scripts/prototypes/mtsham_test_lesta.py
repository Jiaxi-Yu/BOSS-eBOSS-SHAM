#!/usr/bin/env python3
import time
initial = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
from multiprocessing import Pool 
from itertools import repeat
import glob
import sys

gal      = 'LRG'
GC       = 'NGC'
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
Om = 0.31
home      = '/home/astro/jiayu/Desktop/HAM-MCMC/'

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate separation bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
# generate mu bins
s = (bins[1:]+bins[:-1])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf
print('reading the halo catalogue for creating the galaxy catalogue...')

# multiprocessing
sig = sys.argv[1]
sigV= sys.argv[2]
Vceil = sys.argv[3]
nseed    = sys.argv[4]

halofile = home+'catalog/UNIT4LRG-cut200.dat'
print('reading UNIT catalog')
data = Table.read(halofile,format='ascii.no_header')

# make sure len(data) is even
if len(data)%2==1:
data = data[:-1]

print('selecting only the necessary variables...')
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
datac[:,i] = np.copy(data[key])
datac = datac.astype('float32')
half = int(len(data)/2)
scathalf = int(SHAMnum/2)

# generate nseed Gaussian random number arrays in a list
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=x+1).rand(len(data)).astype('float32') for x in range(nseed)]
uniform_randoms1 = [np.random.RandomState(seed=x+31).rand(len(data)).astype('float32') for x in range(nseed)]
print('the uniform random number dtype is ',uniform_randoms[0].dtype)

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
x00,x20,x40,x001,x201,x401=sham_cal(uni,sigM,sigV,Mtrun)
print('first finished')
x01,x21,x41,x011,x211,x411=sham_cal(uni1,sigM,sigV,Mtrun)
print('second finished')
return  [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2,(x001+x011)/2,(x201+x211)/2,(x401+x411)/2]

def sham_cal(uniform,sigma_high,sigma,v_high):
datav = datac[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
print('scattered')
LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],SHAMnum)[:(SHAMnum)]]
print('selected')
# transfer to the redshift space
z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
z_redshift %=boxsize
# calculate the 2pcf of the SHAM galaxies
# count the galaxy pairs and normalise them
DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
# calculate the 2pcf and the multipoles
mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
DD_counts1 = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],LRGscat[:,2],periodic=True, verbose=True,boxsize=boxsize)
# calculate the 2pcf and the multipoles
mono1 = (DD_counts1['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
quad1 = mono1 * 2.5 * (3 * mu**2 - 1)
hexa1 = mono1 * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
# use sum to integrate over mu
return [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu,np.sum(mono1,axis=-1)/nmu,np.sum(quad1,axis=-1)/nmu,np.sum(hexa1,axis=-1)/nmu]

with Pool(processes = nseed) as p:
xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(np.float32(sig)),repeat(np.float32(sigV)),repeat(np.float32(Vceil))))
xi0,xi2,xi4,xi01,xi21,xi41 = mean(xi0_tmp,axis=0,dtype='float32')[0],mean(xi0_tmp,axis=0,dtype='float32')[1],mean(xi0_tmp,axis=0,dtype='float32')[2],mean(xi0_tmp,axis=0,dtype='float32')[3],mean(xi0_tmp,axis=0,dtype='float32')[4],mean(xi0_tmp,axis=0,dtype='float32')[5]
print('averaged')
if sham=='0':
Table([xi0,xi2,xi4]).write(home+'LRG_NGC-redshift_space-python.dat',format = 'ascii.no_header',delimiter='\t',overwrite=True)
Table([xi01,xi21,xi41]).write(home+'LRG_NGC-real_space-python.dat',format = 'ascii.no_header',delimiter='\t',overwrite=True)
else:
Table([xi0,xi2,xi4]).write(home+'LRG_NGC-redshift_space_sigma{}_sigmaV{}_Vceil{}-python.dat'.format(sig,sigV,Vceil),format = 'ascii.no_header',delimiter='\t',overwrite=True)
Table([xi01,xi21,xi41]).write(home+'LRG_NGC-real_space_sigma{}_sigmaV{}_Vceil{}-python.dat'.format(sig,sigV,Vceil),format = 'ascii.no_header',delimiter='\t',overwrite=True)

    python = Table.read(home+'{}_{}-{}-python.dat'.format(gal,GC,space),format='ascii.no_header')
    c = Table.read(home+'{}_{}-{}-c.dat'.format(gal,GC,space),format='ascii.no_header')
elif:
    python = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil{}-python.dat'.format(gal,GC,space,sig,sigV,Vceil),format='ascii.no_header')
    c = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil1e99-c.dat'.format(gal,GC,space,sig,sigV),format='ascii.no_header')
else:
    python = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil{}-python.dat'.format(gal,GC,space,sig,sigV,Vceil),format='ascii.no_header')
    c = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil{}-c.dat'.format(gal,GC,space,sig,sigV,Vceil),format='ascii.no_header')
    
