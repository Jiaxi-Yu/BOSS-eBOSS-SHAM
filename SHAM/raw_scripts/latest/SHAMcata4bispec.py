import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.wp import wp
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import pymultinest
import corner
import h5py

# variables
gal      = 'LRG'
GC       = 'SGC'
date     = '1118'
cut      = 'indexcut'
nseed    = 20
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 2
rmax     = 20
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home     = '/global/cscratch1/sd/jiaxi/SHAM/'
mode    = 'close_chi2'
func    = 'bispec'

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    ver='v7_2'
    halofile = home+'catalog/UNIT_hlist_0.58760.hdf5' 

if gal == 'ELG':
    SHAMnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    ver='v7'
    halofile = home+'catalog/UNIT_hlist_0.53780.hdf5'
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# create the halo catalogue and plot their 2pcf
print('selecting only the necessary variables...')
f=h5py.File(halofile,"r")
sel = f["halo"]['Vpeak'][:]>200
if len(f["halo"]['Vpeak'][:][sel])%2 ==1:
    datac = np.zeros((len(f["halo"]['Vpeak'][:][sel])-1,5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = (f["halo"][key][:][sel])[:-1]
else:
    datac = np.zeros((len(f["halo"]['Vpeak'][:][sel]),5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = f["halo"][key][:][sel]
f.close()        
half = int(len(datac)/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
seed = [x for x in range(nseed)]
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 
seed1 = [x+30 for x in range(nseed)]

# HAM application
def sham_tpcf(uni,uni1,seed,seed1,sigM,sigV,Mtrun):      
    x00    = sham_cal(uni,seed,sigM,sigV,Mtrun)
    x01    = sham_cal(uni1,seed1,sigM,sigV,Mtrun)
    return None

def sham_cal(uniform,seednum,sigma_high,sigma,v_high):
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
        
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
    LRGscat[:,-1]  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    LRGscat[:,-1] %=boxsize
    np.savetxt('testdata/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,sigma_high,int(sigma),int(10**v_high),seednum),LRGscat[:,2:])
    return None

# calculate the SHAM 2PCF
with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,seed,seed1,repeat(np.float32(0.14864245)),repeat(131.20083923),repeat(np.float32(5.24729866))))) 
    
with Pool(processes = nseed) as p:
    xi0_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,seed,seed1,repeat(np.float32(0.59092776)),repeat(109.82603879),repeat(np.float32(4.87587909))))) 
