#!/usr/bin/env python3
import time
initial = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack
from astropy.table import Table
from astropy.io import fits
import os
from multiprocessing import Pool 
import glob
import sys

# variables
function = 'mps'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
rscale = 'log'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 30
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home      = '/global/cscratch1/sd/jiaxi/master/'

for gal in ['LRG','ELG']:
    for GC in ['NGC','SGC','NGC+SGC']:
        # read s bins
        binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
        bins  = np.unique(np.append(binfile['col1'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)],binfile['col2'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]))
        binmin = np.where(binfile['col3']>=rmin)[0][0]
        binmax = np.where(binfile['col3']<rmax)[0][-1]+1
        if (gal == 'LRG'):
            ver = 'v7_2'
            extra = binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]**2
            Zmin = [0.6,0.6,0.65,0.7,0.8]
            Zmax = [0.7,0.8,0.8,0.9,1.0]
        else:
            ver = 'v7'
            extra = np.ones(binmax-binmin)
            Zmin = [0.6,0.9,0.7,0.8]
            Zmax = [0.8,1.1,0.9,1.0]
        for zmin,zmax in zip(Zmin,Zmax):
            # zbins with log binned mps and wp
            covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
            covR = '{}catalog/nersc_zbins_wp_mps_{}/covR_{}_{}_{}_z{}z{}_mocks_{}.dat'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
            obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
            # Read the covariance matrices and observations
            hdu = fits.open(covfits) # cov([mono,quadru])
            mocks = hdu[1].data[GC+'mocks']
            Nmock = mocks.shape[1] 
            errbar = np.std(mocks,axis=1)
            hdu.close()
            obscf = Table.read(obs2pcf,format='ascii.no_header')
            # LRG columns are s**2*xi
            obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
            nbins = len(bins)-1
            s = obscf['col3']
            # inverse covariance
            if multipole=='quad':
                Ns = int(mocks.shape[0]/2)
                mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
                covcut  = cov(mocks).astype('float32')
                OBS   = append(obscf['col4']/extra,obscf['col5']/extra).astype('float32')
                Nbins = len(OBS)
                COVR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)  

            np.savetxt(covR,COVR)
            if gal=='LRG':
                OBS.write('{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}_nos.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax),format = 'ascii.no_header',delimiter='\t',overwrite=True)

