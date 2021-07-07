#!/usr/bin/env python3
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

binmin=5
binmax=25
gal  = 'LRG'
function  = sys.argv[1] # '2PCF'  'wp'
npi = bool(int(sys.argv[2]))
home     = '/home/astro/jiayu/Desktop/SHAM/'
datapath = home+'catalog/wp_diff-pi/'
ver = 'v7'
Zrange =  np.array([0.6,1.0])
znum = int(len(Zrange)/2)

if function == '2PCF':
    rscale = 'linear'
    nbins = 120
    nmu = 120
    func = 'mps'
    s = (np.arange(0,120)+np.arange(1,121))/2
elif function == 'wp':
    rscale = 'log'
    nbins  = 8
    nmu    = 80
    func   = 'wp'
    pimaxs  = [15,25,60,80]
    swp = np.loadtxt(home+'binfile_log.dat',usecols=2)
    s = swp[(swp>binmin)&(swp<binmax+1)]


#######################
# plot mocks
for mocktype in ['sys','nosys']:
    mockDIR  = ['/hpcstorage/jiayu/EZmocks_{}/z{}z{}/clustering/'.format(mocktype,Zrange[k],Zrange[k+znum]) for k in range(znum)]
    mockFITS  = ['{}EZmocks_{}_{}_{}_z{}z{}_'.format(datapath,mocktype,function,rscale,Zrange[k],Zrange[k+znum]) for k in range(znum)]
    for k,mockdir,mockfits in zip(range(znum),mockDIR,mockFITS): 
        start=time.time()
        ddpath = glob.glob(mockdir+'*_SGC'+'*.dd')
        nfile = len(ddpath)
        # read all the 2pcf data
        mockmono = [x for x in range(nfile)]
        mockquad = [x for x in range(nfile)]
        mockhexa = [x for x in range(nfile)]

        if (function == '2PCF')&(not os.path.exists(mockFITS[-1]+'quad.fits.gz')):
            pairroot = [mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}' for n in range(nfile)]
            # NGC+SGC for mono, quad, hexadeca-pole
            if mocktype == 'sys':
                comb_part = partial(FCFCcomb,ns=nbins,nmu=nmu)
            else:
                rrroot = mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr'
                comb_part = partial(FCFCcomb,rfmt = rrroot,ns=nbins,nmu=nmu)

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
                print(name,' saved')
                hdulist = fits.BinTableHDU.from_columns(cols)
                hdulist.header.update(sbins=nbins,nmu=nmu)
                hdulist.writeto(mockfits+name+'.fits.gz',overwrite=True)
        elif (function == 'wp')&(not os.path.exists(mockFITS[-1]+'wp_pi80.fits.gz'))&(not npi):
            pairroot = [mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}.wp' for n in range(nfile)]
            if mocktype == 'sys':
                comb_part = partial(FCFCcomb,ns=nbins,nmu=nmu,islog=True,ismps=False)
            else:
                rrroot = mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr.wp'
                comb_part = partial(FCFCcomb,rfmt = rrroot,ns=nbins,nmu=nmu,islog=True,ismps=False)

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
            hdulist.writeto(mockfits+'wp_pi80.fits.gz',overwrite=True)
        elif (function == 'wp')&(os.path.exists(mockFITS[-1]+'wp_pi80.fits.gz'))&(npi):
            #"""
            pairroot = [mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'_'+str(n+1).zfill(4)+'.{}.wp' for n in range(nfile)]
            for pimax in pimaxs[:-1]:
                if os.path.exists(mockFITS[-1]+'wp_pi{}.fits.gz'.format(pimax)):
                    pass    
                else: 
                    if mocktype == 'sys':
                        comb_part = partial(FCFCcomb,ns=nbins,nmu=nmu,islog=True,ismps=False)
                    else:
                        rrroot = mockdir+'EZmock_eBOSS_{}_'+ver+'_z'+str(Zrange[k])+'z'+str(Zrange[k+znum])+'.rr.wp'
                        comb_part = partial(FCFCcomb,rfmt = rrroot,ns=nbins,nmu=nmu,upperint=pimax,islog=True,ismps=False)


                    # NGC+SGC for wp
                    pool = Pool()     
                    for n, temp_array in enumerate(pool.imap(comb_part,pairroot)):
                        mockmono[n] = temp_array
                        if n==0:
                            print('mock reading&2PCF calculation start')
                        elif (n+1)%(nfile/10)==0:
                            print('wp_pi{} of {} {} bin PATCHY mocks at {}<z<{} has finished {}%'.format(pimax,gal,rscale,Zrange[k],Zrange[k+znum],(n+1)//(nfile/100)))
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
            #""" 
        fin=time.time()
        print('z{}z{} NGC+SGC and covariance calculation finished in {:.6}s'.format(Zrange[k],Zrange[k+znum],-start+fin))


# plot 2pcf comparison
if os.path.exists(mockFITS[-1]+'quad.fits.gz'):
    hdu = fits.open('{}EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1])) #
    mocks = hdu[1].data['NGC+SGCmocks']
    Nmock = mocks.shape[1] 
    hdu.close()
    errbarsys = np.std(mocks,axis=1)
    meansys = np.mean(mocks,axis=1)

    hdu = fits.open('{}EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1])) #
    mocks = hdu[1].data['NGC+SGCmocks']
    Nmock = mocks.shape[1] 
    hdu.close()
    errbarnosys = np.std(mocks,axis=1)
    meannosys = np.mean(mocks,axis=1)

    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.2)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for name,k in zip(['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),meansys[k*nbins:(k+1)*nbins]]        
        err   = [np.ones(nbins),s**2*errbarsys[k*nbins:(k+1)*nbins]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,s**2*(meansys[k*nbins:(k+1)*nbins]-values[j])/err[j],color='r', label='EZmock_sys')
            ax[j,k].plot(s,s**2*(meannosys[k*nbins:(k+1)*nbins]-values[j])/err[j],color='b', label='EZmock_nosys')
            ax[j,k].fill_between(s,s**2*((meansys-errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meansys+errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color='r',alpha=0.4)
            ax[j,k].fill_between(s,s**2*((meannosys-errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meannosys+errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color='b',alpha=0.4)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                if k==0:
                    pass
                    #plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('correlation function {}'.format(name))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2))
                plt.ylim(-3,3)

    plt.savefig('mps_sys-vs-nosys.png')
    plt.close()

# plot wp comparison
if os.path.exists(mockFITS[-1]+'wp_pi80.fits.gz'):
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(7*len(pimaxs),8))
    spec = gridspec.GridSpec(nrows=2,ncols=len(pimaxs), height_ratios=[4, 1], hspace=0.3,wspace=0.2)
    ax = np.empty((2,len(pimaxs)), dtype=type(plt.axes))
    for k,pimax in enumerate(pimaxs):
        hdu = fits.open('{}EZmocks_{}_{}_{}_z{}z{}_wp_pi{}.fits.gz'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1],pimax)) #
        mocks = hdu[1].data['NGC+SGCmocks']
        Nmock = mocks.shape[1] 
        hdu.close()
        errbarsys = np.std(mocks,axis=1)
        meansys = np.mean(mocks,axis=1)

        hdu = fits.open('{}EZmocks_{}_{}_{}_z{}z{}_wp_pi{}.fits.gz'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1],pimax)) #
        mocks = hdu[1].data['NGC+SGCmocks']
        Nmock = mocks.shape[1] 
        hdu.close()
        errbarnosys = np.std(mocks,axis=1)
        meannosys = np.mean(mocks,axis=1)

        # plot wp with different pi
        values=[np.zeros(nbins),meansys]        
        err   = [np.ones(nbins),errbarsys]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,(meansys-values[j])/err[j],color='r', label='EZmock_sys')
            ax[j,k].plot(s,(meannosys-values[j])/err[j],color='b', label='EZmock_nosys')
            ax[j,k].fill_between(s,((meansys-errbarsys)-values[j])/err[j],((meansys+errbarsys)-values[j])/err[j],color='r',alpha=0.4)
            ax[j,k].fill_between(s,((meannosys-errbarnosys)-values[j])/err[j],((meannosys+errbarnosys)-values[j])/err[j],color='b',alpha=0.4)
            plt.xlabel('$r_p$ (Mpc $h^{-1}$)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('wp')#('\\xi_{}$'.format(k*2))#
                if k==0:
                    plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('wp($\pi$ in [0,{}]'.format(pimax)+' Mpc$h^{-1}$)')
            if (j==1):
                ax[j,k].set_ylabel('$\Delta$ wp/err')
                plt.ylim(-3,3)

    plt.savefig('wp_sys-vs-nosys.png')
    plt.close()