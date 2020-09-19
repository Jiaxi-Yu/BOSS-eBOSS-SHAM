# calculate the covariance of the mock 2pcf 
# save both the 1000 mock 2pcf and its covariance
# and calculate the 2pcf of observations
import time
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
from NGC_SGC import read_xi
import sys


Om = 0.31
home      = '/global/cscratch1/sd/jiaxi/master/'
function  = 'mps' #not yet
gal  = sys.argv[1]
rscale = sys.argv[2] #'linear' # 'log'

if (rscale == 'linear'):
    k=0
    if gal=='ELG':
        Zrange =  [0.6,\
                   1.1]
        # EZmocks directory
        nbins=200
        nmu=120
        znum = int(len(Zrange)/2)
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z0.6z1.1/2PCF/'
        mockfits  = home+'catalog/nersc_mps_ELG_v7/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'  
    else:
        Zrange =  [0.6,\
                   1.0]
        nbins=200
        nmu=120
        znum = int(len(Zrange)/2)
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/'
        mockfits  = home+'catalog/nersc_mps_LRG_v7_2/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'   
    # covfits
    ddpath = glob.glob(mockdir+'*'+gal+'_NGC'+'*.dd')
    nfile = len(ddpath)
    # read all the 2pcf data
    mockmono = [x for x in range(nfile)]
    mockquad = [x for x in range(nfile)]
    mockhexa = [x for x in range(nfile)]
    for n in range(nfile):
        s, mockmono[n],mockquad[n],mockhexa[n] = read_xi(mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}',ns=nbins)
        if n==0:
            print('Ezmock reading&2PCF calculation start')
        elif (n+1)%100==0:
            print('{} {} bin EZmock at {}<z<{} has finished {}%'.format(gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//10))
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

elif (rscale == 'log'):
    nbins=len(ascii.read(home+'binfile_log.dat'))
    nmu=120
    if gal=='ELG':
        Zrange =  np.array([0.6,0.7,0.8,0.9,\
                   0.8,0.9,1.0,1.1])
    else:
        Zrange =  np.array([0.6,0.7,0.8,0.6,0.65,\
                   0.8,0.9,1.0,0.7,0.8])
    znum = int(len(Zrange)/2)
    mockDIR  = ['{}catalog/nersc_zbins_wp_mps_{}/EZmocks/z{}z{}/2PCF/'.format(home,gal,Zrange[k],Zrange[k+znum]) for k in range(znum)]
    mockFITS  = ['{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_'.format(home,gal,function,rscale,gal,Zrange[k],Zrange[k+znum]) for k in range(znum)]

    for k,mockdir,mockfits in zip(range(znum),mockDIR,mockFITS): 
        time_start=time.time()
        ddpath = glob.glob(mockdir+'*'+gal+'_NGC'+'*.dd')
        nfile = len(ddpath)
        # read all the 2pcf data
        mockmono = [x for x in range(nfile)]
        mockquad = [x for x in range(nfile)]
        mockhexa = [x for x in range(nfile)]
        for n in range(nfile):
            s, mockmono[n],mockquad[n],mockhexa[n] = read_xi(mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}',mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr',ns=nbins)
            if n==0:
                print('Ezmock reading&2PCF calculation start')
            elif (n+1)%100==0:
                print('{} {} bin EZmock at {}<z<{} has finished {}%'.format(gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//10))
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

