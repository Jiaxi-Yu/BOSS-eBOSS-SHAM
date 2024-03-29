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

#binmin=5
#binmax=25
binmin=0
binmax=100
ds = 5
gal  = sys.argv[1]
function  = sys.argv[2] # '2PCF'  'wp'
if function == '2PCF':
    rscale = 'linear'
    nbins = 100
    nmu = 120
    func = 'mps'
elif function == 'wp':
    rscale = 'log'
    nbins  = 8
    nmu    = 80
    func   = 'wp'
    pimaxs  = [25,30,35]

home     = '/home/astro/jiayu/Desktop/SHAM/'
datapath = '{}catalog/BOSS_zbins_{}/'.format(home,func)
ver = 'DR12'

# z range, version and datapath
if gal=='CMASS':
    Zrange =  np.array([0.43,\
                        0.7]) #0.43,0.51,0.57, #0.51,0.57,0.7,
elif gal == 'LOWZ':
    Zrange =  np.array([0.2, 0.33,0.2,\
                        0.33,0.43,0.43])
elif gal == 'CMASSLOWZTOT':
    Zrange = np.array([0.2,0.4,\
                       0.5,0.6])
    ver = 'DR12v5'
    nbins = 120
elif gal == 'LRG':
    Zrange =  np.array([0.6,0.6,0.65,0.7,0.8,0.6,\
                        0.7,0.8,0.8, 0.9,1.0,1.0])
    # no 2PCF combination for EZmocks
    function = 'wp'
    ver = 'v7'
    datapath = '{}catalog/nersc_zbins_wp_mps_{}/'.format(home,gal)
znum = int(len(Zrange)/2)

#######################
# mock paircount directory root
if gal == 'LRG':
    mockDIR  = ['/hpcstorage/jiayu/EZmocks_sys/z{}z{}/wp/'.format(Zrange[k],Zrange[k+znum]) for k in range(znum)]
else:
    mockDIR  = ['/hpcstorage/jiayu/PATCHY/{}/z{}z{}/{}/'.format(gal,Zrange[k],Zrange[k+znum],function) for k in range(znum)]

# output root
#mockFITS  = ['{}{}_{}_z{}z{}_mocks_'.format(datapath,gal,rscale,Zrange[k],Zrange[k+znum]) for k in range(znum)]
mockFITS  = ['{}_{}_z{}z{}_mocks_'.format(gal,rscale,Zrange[k],Zrange[k+znum]) for k in range(znum)]

for k,mockdir,mockfits in zip(range(znum),mockDIR,mockFITS): 
    start=time.time()
    ddpath = glob.glob(mockdir+'*_SGC'+'*.dd')
    nfile = len(ddpath)
    # read all the 2pcf data
    mockmono = [x for x in range(nfile)]
    mockquad = [x for x in range(nfile)]
    mockhexa = [x for x in range(nfile)]
    # paircount root: dd(dr) and rr
    if gal == 'LRG':
        pairroot = [mockdir+'wp_EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}' for n in range(nfile)]
    else:
        pairroot = [mockdir+function+'_PATCHYmock_'+gal+'_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}' for n in range(nfile)]
        rrfile   = mockdir+function+'_PATCHYmock_'+gal+'_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr'
    # no 0009 for CMASS PATCHY SGC
    if gal == 'CMASS':
        pairroot[9] = mockdir+function+'_PATCHYmock_'+gal+'_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_1200.{}'

    if function == '2PCF':
        if ds ==1:
            mockfile = mockfits+'hexa.fits.gz'
        else:
            mockfile = mockfits+'hexa_bin{}.fits.gz'

        if os.path.exists(mockfile):
            pass
        else:
            comb_part = partial(FCFCcomb,rfmt = rrfile,ds=ds)
            # NGC+SGC for mono, quad, hexadeca-pole
            pool = Pool()     
            for n, temp_array in enumerate(pool.imap(comb_part,pairroot)):
                mockmono[n],mockquad[n],mockhexa[n]= temp_array
                if n==0:
                    print('mock reading&2PCF calculation start')
                elif (n+1)%(np.ceil(nfile/10))==0:
                    print('{} {} bin mocks at {}<z<{} has finished {}%'.format(gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//(nfile/100)))
            pool.close() 
            pool.join()
            print('finished')
            # stack mono, mono+quad, mono+quad+hexa
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
            for j,name in enumerate(['mono','quad','hexa']):
                cols = []
                cols.append(fits.Column(name='NGCmocks',format=str(nfile)+'D',array=NGC[j]))
                cols.append(fits.Column(name='SGCmocks',format=str(nfile)+'D',array=SGC[j]))
                cols.append(fits.Column(name='NGC+SGCmocks',format=str(nfile)+'D',array=NGCSGC[j]))
                print(name,' saved',NGCSGC[j].shape)
                hdulist = fits.BinTableHDU.from_columns(cols)
                hdulist.header.update(sbins=nbins,nmu=nmu)
                if ds ==1:
                    hdulist.writeto(mockfits+name+'.fits.gz',overwrite=True)
                else:
                    hdulist.writeto(mockfits+name+'_bin{}.fits.gz'.format(ds),overwrite=True)
            print(mockfits+name+'_bin{}.fits.gz'.format(ds))
            if ds ==1:
                # covR calculation for 5-25Mpc/h & quadrupole
                hdu = fits.open(mockfits+'quad.fits.gz')
                for GC in ['NGC+SGC']:
                    mocks = hdu[1].data[GC+'mocks'] 
                    Nmock = mocks.shape[1] 
                    Ns = int(mocks.shape[0]/2)
                    mocks = np.vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
                    covcut  = np.cov(mocks).astype('float32')
                    Nbins = len(mocks)
                    covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
                    np.savetxt(datapath+'covR-'+gal+'_'+GC+'-s'+str(binmin)+'_'+str(binmax)+'-z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'-quad.dat',covR)
                hdu.close()

        fin=time.time()
        print('z{}z{} NGC+SGC and covariance calculation finished in {:.6}s'.format(Zrange[k],Zrange[k+znum],-start+fin))
    elif function == 'wp':
        if os.path.exists(mockfits+'wp.fits.gz'):
            pass
        else:
            if gal == 'LRG':
                comb_part = partial(FCFCcomb,ns=nbins,nmu=nmu,islog=True,ismps=False)
            else:    
                comb_part = partial(FCFCcomb,rfmt = rrfile,ns=nbins,nmu=nmu,islog=True,ismps=False)

            # NGC+SGC for wp
            pool = Pool()     
            for n, temp_array in enumerate(pool.imap(comb_part,pairroot)):
                mockmono[n] = temp_array
                if n==0:
                    print('mock reading&2PCF calculation start')
                elif (n+1)%(nfile/10)==0:
                    print('{} {} bin mocks at {}<z<{} has finished {}%'.format(gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//(nfile/100)))
            pool.close() 
            pool.join()

            # stack mono, mono+quad, mono+quad+hexa
            NGC  = np.array([mockmono[k][0] for k in range(nfile)]).T
            SGC  = np.array([mockmono[k][1] for k in range(nfile)]).T
            NGCSGC = np.array([mockmono[k][2] for k in range(nfile)]).T

            # save data as binary table
            # name of the mock 2pcf and covariance matrix file(function return)
            cols = []
            cols.append(fits.Column(name='NGCmocks',format=str(nfile)+'D',array=NGC))
            cols.append(fits.Column(name='SGCmocks',format=str(nfile)+'D',array=SGC))
            cols.append(fits.Column(name='NGC+SGCmocks',format=str(nfile)+'D',array=NGCSGC))

            hdulist = fits.BinTableHDU.from_columns(cols)
            hdulist.header.update(sbins=nbins,nmu=nmu)
            hdulist.writeto(mockfits+'wp.fits.gz',overwrite=True)

            # wp for different max_pi
            for pimax in pimaxs:
                if os.path.exists(mockfits+'pi{}.fits.gz'.format(pimax)):
                    pass
                else:
                    if gal == 'LRG':
                        comb_part = partial(FCFCcomb,ns=nbins,nmu=nmu,upperint=pimax,islog=True,ismps=False)
                    else:    
                        comb_part = partial(FCFCcomb,rfmt = rrfile,ns=nbins,nmu=nmu,upperint=pimax,islog=True,ismps=False)

                    # NGC+SGC for wp
                    pool = Pool()     
                    for n, temp_array in enumerate(pool.imap(comb_part,pairroot)):
                        mockmono[n] = temp_array
                        if n==0:
                            print('mock reading&2PCF calculation start')
                        elif (n+1)%(nfile/10)==0:
                            print('wp_pi{} of {} {} bin mocks at {}<z<{} has finished {}%'.format(pimax,gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//(nfile/100)))
                    pool.close() 
                    pool.join()

                    # stack mono, mono+quad, mono+quad+hexa
                    NGC  = np.array([mockmono[k][0] for k in range(nfile)]).T
                    SGC  = np.array([mockmono[k][1] for k in range(nfile)]).T
                    NGCSGC = np.array([mockmono[k][2] for k in range(nfile)]).T

                    # save data as binary table
                    # name of the mock 2pcf and covariance matrix file(function return)
                    cols = []
                    cols.append(fits.Column(name='NGCmocks',format=str(nfile)+'D',array=NGC))
                    cols.append(fits.Column(name='SGCmocks',format=str(nfile)+'D',array=SGC))
                    cols.append(fits.Column(name='NGC+SGCmocks',format=str(nfile)+'D',array=NGCSGC))

                    hdulist = fits.BinTableHDU.from_columns(cols)
                    hdulist.header.update(sbins=nbins,nmu=nmu,pimax=pimax)
                    hdulist.writeto('{}wp_pi{}.fits.gz'.format(mockfits,pimax),overwrite=True)


"""
############
# observations:
obsfiles  = [datapath+'OBS_'+gal+'_{}_DR12v5_'+'z{}z{}'.format(Zrange[k],Zrange[k+znum]) for k in range(znum)]

for k,obsfile in zip(range(znum),obsfiles): 
    obsxi0,obsxi2,obsxi4 = FCFCcomb(obsfile+'.{}',obsfile+'.rr',ns=nbins,nmu=nmu,islog=True,isobs=True,ismps=False)
"""