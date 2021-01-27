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
from NGC_SGC_wp import cal_wp
import sys


Om = 0.31
home      = '/global/cscratch1/sd/jiaxi/SHAM/'
binmin=5
binmax=25
gal  = sys.argv[1]
rscale = sys.argv[2] #'linear' # 'log'
function  = 'mps' 


if (rscale == 'linear'):
    k=0
    if gal=='ELG':
        Zrange =  [0.6,\
                   1.1]
        # EZmocks directory
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z0.6z1.1/2PCF/'
        mockfits  = home+'catalog/nersc_mps_ELG_v7/'+function+'_'+rscale+'_'+gal+'_mocks_'  
        file = 'nersc_mps_ELG_v7'
    else:
        Zrange =  [0.6,\
                   1.0]
        mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/'
        mockfits  = home+'catalog/nersc_mps_LRG_v7_2/'+function+'_'+rscale+'_'+gal+'_mocks_'   
        file = 'nersc_mps_LRG_v7_2'
    nbins=200
    nmu=120
    znum = int(len(Zrange)/2)
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

    # covR calculation for 5-25Mpc/h & quadrupole
    hdu = fits.open(home+'catalog/'+file+'/'+function+'_'+rscale+'_'+gal+'_mocks_quad.fits.gz')
    for GC in ['NGC','SGC','NGC+SGC']:
        mocks = hdu[1].data[GC+'mocks'] 
        Nmock = mocks.shape[1] 
        Ns = int(mocks.shape[0]/2)
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
        covcut  = cov(mocks).astype('float32')
        Nbins = len(mocks)
        covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
        np.savetxt(home+'catalog/'+file+'/covR-'+gal+'_'+GC+'-5_25-quad.txt',covR)
    hdu.close()
elif (rscale=='log')&(function=='mps'):
    nbins=len(ascii.read(home+'binfile_log.dat'))
    k=0
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
            mockmono[n],mockquad[n],mockhexa[n] = read_xi(mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}',mockdir+'2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr',ns=nbins)
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

        # covR calculation for 5-25Mpc/h & quadrupole
        hdu = fits.open(mockfits+'quad.fits.gz')
        for GC in ['NGC','SGC','NGC+SGC']:
            mocks = hdu[1].data[GC+'mocks'] 
            Nmock = mocks.shape[1] 
            Ns = int(mocks.shape[0]/2)
            mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
            covcut  = cov(mocks).astype('float32')
            Nbins = len(mocks)
            covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
            np.savetxt(mockfits[:-23]+'covR-'+gal+'_'+GC+'-s'+str(binmin)+'_'+str(binmax)+'-z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'-quad.dat',covR)
        hdu.close()
else:
    if gal=='LRG':
        mockdir = '{}catalog/nersc_wp_{}_{}/EZmocks/EZmock_{}_'.format(home,gal,'v7_2',gal)
        ver = 'v7_2'
    else:
        mockdir = '{}catalog/nersc_wp_{}_{}/EZmocks/EZmock_{}_'.format(home,gal,'v7',gal)
        ver = 'v7'
    mockfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(home,function,gal,ver,function,gal) 
    # covfits
    ddpath = glob.glob(mockdir+'NGC'+'*.dat')
    nfile = len(ddpath)
    # read all the 2pcf data
    mockmono = [x for x in range(nfile)]
    ns = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')['col3']
    num = np.loadtxt(home+'EZmocks_{}_Ngal.dat'.format(gal))
    for n in range(nfile):
        wp, mockmono[n] = cal_wp(mockdir+'{}_'+str(n+1).zfill(4)+'.dat',len(ns),num[n,:])
        if n==0:
            print('Ezmock reading&2PCF calculation start')
        elif (n+1)%100==0:
            print('{} {} bin EZmock has finished {}%'.format(gal,rscale,(n+1)//10))
    print('all finished')

    # calculate the covariance
    NGC  = np.array([mockmono[k][0] for k in range(nfile)]).T
    SGC  = np.array([mockmono[k][1] for k in range(nfile)]).T
    """
    NGCSGC = [np.array([mockmono[k][2] for k in range(nfile)]).T]
    """
    # save data as binary table
    # name of the mock 2pcf and covariance matrix file(function return)
    cols = []
    cols.append(fits.Column(name='NGCmocks',format=str(nfile)+'D',array=NGC))
    cols.append(fits.Column(name='SGCmocks',format=str(nfile)+'D',array=SGC))
    #cols.append(fits.Column(name='NGC+SGCmocks',format=str(nfile)+'D',array=NGCSGC[k]))

    hdulist = fits.BinTableHDU.from_columns(cols)
    hdulist.header.update(sbins=len(NGC),nmu=80)
    hdulist.writeto(mockfits,overwrite=True)
"""
    # covR calculation for 5-25Mpc/h & quadrupole
    hdu = fits.open(home+'catalog/'+file+'/'+function+'_'+rscale+'_'+gal+'_mocks_quad.fits.gz')
    for GC in ['NGC','SGC']:#,'NGC+SGC']:
        mocks = hdu[1].data[GC+'mocks'] 
        Nmock = mocks.shape[1] 
        Ns = int(mocks.shape[0]/2)
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
        covcut  = cov(mocks).astype('float32')
        Nbins = len(mocks)
        covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
        np.savetxt(home+'catalog/'+file+'/covR-'+gal+'_'+GC+'-5_25-quad.txt',covR)
    hdu.close()
    
time_end=time.time()
print('Covariance matrix calculation costs',time_end-time_start,'s')

"""
