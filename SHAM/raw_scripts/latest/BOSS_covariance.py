import time
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from NGC_SGC import FCFCcomb
from multiprocessing import Pool
from functools import partial
import os
import glob
import sys
import re

Om = 0.31
home     = '/home/astro/jiayu/Desktop/SHAM/'
datapath = home+'catalog/BOSS_zbins_mps/'
binmin=5
binmax=25
gal  = sys.argv[1]
rscale = 'linear'
function  = 'mps' 
nbins = 100
k=0
nmu=120
if gal=='CMASS':
    Zrange =  np.array([0.43,0.51,0.57,0.43,\
                        0.51,0.57,0.7, 0.7])
elif gal == 'LOWZ':
    Zrange =  np.array([0.2, 0.33,0.2,\
                        0.33,0.43,0.43])
znum = int(len(Zrange)/2)
mockDIR  = ['/hpcstorage/jiayu/PATCHY/{}_1200/z{}z{}/2PCF/'.format(gal,Zrange[k],Zrange[k+znum]) for k in range(znum)]
mockFITS  = ['{}{}_{}_z{}z{}_mocks_'.format(datapath,gal,rscale,Zrange[k],Zrange[k+znum]) for k in range(znum)]

for k,mockdir,mockfits in zip(range(znum),mockDIR,mockFITS): 
    start=time.time()
    ddpath = glob.glob(mockdir+'*'+gal+'_SGC'+'*.dd')
    nfile = len(ddpath)
    # read all the 2pcf data
    mockmono = [x for x in range(nfile)]
    mockquad = [x for x in range(nfile)]
    mockhexa = [x for x in range(nfile)]
    pairroot = [mockdir+'2PCF_PATCHYmock_'+gal+'_{}_DR12_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}' for n in range(nfile)]
    rrfile   = mockdir+'2PCF_PATCHYmock_'+gal+'_{}_DR12_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr')
    comb_part = partial(FCFCcomb,rfmt = rrfile)

    pool = Pool()     
    for n, temp_array in enumerate(pool.imap(comb_part,pairroot)):
        mockmono[n],mockquad[n],mockhexa[n]= temp_array
        if n==0:
            print('Ezmock reading&2PCF calculation start')
        elif (n+1)%100==0:
            print('{} {} bin PATCHY mocks at {}<z<{} has finished {}%'.format(gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//10))
    pool.close() 
    pool.join()

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
        np.savetxt(datapath+'covR-'+gal+'_'+GC+'-s'+str(binmin)+'_'+str(binmax)+'-z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'-quad.dat',covR)
    hdu.close()
    fin=time.time()
    print('z{}z{} NGC+SGC and covariance calculation finished in {:.6}s'.format(Zrange[k],Zrange[k+znum],-start+fin))
