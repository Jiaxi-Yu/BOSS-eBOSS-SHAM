# calculate the covariance of the mock 2pcf 
# save both the 1000 mock 2pcf and its covariance
# and calculate the 2pcf of observations
import time
time_start=time.time()
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
from NGC_SGC import read_xi

Om = 0.31
home      = '/global/cscratch1/sd/jiaxi/master/'
function  = 'mps'
rscale = 'linear'


for gal in ['LRG','ELG']:
    # EZmocks directory
    if gal == 'ELG':
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z0.6z1.1/2PCF/'
        zrange='z0.6z1.1'
        if rscale == 'linear':
            mockfits  = home+'catalog/nersc_mps_ELG_v7/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'    
        else:
            mockfits  = home+'catalog/nersc_zbins_wp_mps_ELG/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'  
    if gal == 'LRG':
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/'
        zrange='z0.6z1.0'            
        if rscale == 'linear':
            mockfits  = home+'catalog/nersc_mps_LRG_v7_2/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'   
        else:
            mockfits  = home+'catalog/nersc_zbins_wp_mps_LRG/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_' 

    nbins=200
    nmu=120
    ddpath = glob.glob(mockdir+'*'+gal+'_NGC'+'*.dd')
    nfile = len(ddpath)
    # read all the 2pcf data
    mockmono = [x for x in range(nfile)]
    mockquad = [x for x in range(nfile)]
    mockhexa = [x for x in range(nfile)]
    print('Ezmock reading&2PCF calculation start')
    for n in range(nfile):
        s, mockmono[n],mockquad[n],mockhexa[n] = read_xi(mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_'+zrange+'_'+str(n+1).zfill(4)+'.{}',ds=1)
        if (n+1)%100==0:
            print('{} EZmock has finished {}%'.format(gal,(n+1)//10))
    print('all finished')

    # calculate the covariance
    NGC  = [np.array([mockmono[k][0] for k in range(nfile)]).T,\
            np.vstack((np.array([mockmono[k][0] for k in range(nfile)]).T,\
                       np.array([mockquad[k][0] for k in range(nfile)]).T)),\
            np.vstack((np.array([mockmono[k][0] for k in range(nfile)]).T,\
                       np.array([mockquad[k][0] for k in range(nfile)]).T,\
                       np.array([mockhexa[k][0] for k in range(nfile)]).T))]
    SGC  = [np.array([mockmono[k][1] for k in range(nfile)]).T,\
            np.vstack((np.array([mockmono[k][1] for k in range(nfile)]).T,\
                       np.array([mockquad[k][1] for k in range(nfile)]).T)),\
            np.vstack((np.array([mockmono[k][1] for k in range(nfile)]).T,\
                       np.array([mockquad[k][1] for k in range(nfile)]).T,\
                       np.array([mockhexa[k][1] for k in range(nfile)]).T))]
    NGCSGC = [np.array([mockmono[k][2] for k in range(nfile)]).T,\
            np.vstack((np.array([mockmono[k][2] for k in range(nfile)]).T,\
                       np.array([mockquad[k][2] for k in range(nfile)]).T)),\
            np.vstack((np.array([mockmono[k][2] for k in range(nfile)]).T,\
                       np.array([mockquad[k][2] for k in range(nfile)]).T,\
                       np.array([mockhexa[k][2] for k in range(nfile)]).T))]

    # save data as binary table
    # name of the mock 2pcf and covariance matrix file(function return)
    for k,name in enumerate(['mono','quad','hexa']):
        cols = []
        cols.append(fits.Column(name='NGCmocks',format=str(nfile)+'D',array=NGC[k]))
        cols.append(fits.Column(name='SGCmocks',format=str(nfile)+'D',array=SGC[k]))
        cols.append(fits.Column(name='NGC+SGCmocks',format=str(nfile)+'D',array=NGCSGC[k]))

        hdulist = fits.BinTableHDU.from_columns(cols)
        hdulist.header.update(sbins=nbins,nmu=nmu)
        hdulist.writeto(mockfits+name+'.fits.gz',overwrite=True)

    time_end=time.time()
    print('Covariance matrix calculation costs',time_end-time_start,'s')

