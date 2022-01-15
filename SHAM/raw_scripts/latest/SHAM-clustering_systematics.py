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
#function  = sys.argv[1] # '2PCF'  'wp'
home     = '/home/astro/jiayu/Desktop/SHAM/'
datapath = home+'catalog/wp_diff-pi/'
ver = 'v7_2'
Zrange =  np.array([0.6,1.0])
znum = int(len(Zrange)/2)
colors = ['m','b','orange','r','c']

fontsize=15
plt.rc('font', family='serif', size=fontsize)
fig = plt.figure(figsize=(8,8))
spec = gridspec.GridSpec(nrows=4,ncols=2, left = 0.13,right = 0.98,bottom=0.08,top = 0.98,height_ratios=[2, 1,2,1], hspace=0,wspace=0.3)
ax = np.empty((4,2), dtype=type(plt.axes))

for function in ['2PCF','wp']:
    if function == '2PCF':
        rscale = 'linear'
        nbins = 120
        nmu = 120
        func = 'mps'
        s = (np.arange(0,120)+np.arange(1,121))/2
        M=1
        K=0
        # 2PCF data reading
        hdu = fits.open('{}nosys_FKP/EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1])) #
        mocks = hdu[1].data['NGC+SGCmocks']
        Nmock = mocks.shape[1] 
        hdu.close()
        errbarnosys = np.std(mocks,axis=1)/np.sqrt(Nmock)
        meannosys = np.mean(mocks,axis=1)
        mockweight = '_FKPSYSTOT'
        for name,k in zip(['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),meannosys[k*nbins:(k+1)*nbins]]        
            err   = [np.ones(nbins),s**2*errbarnosys[k*nbins:(k+1)*nbins]]
            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                # nosys: black & grey shade
                ax[J,K].plot(s,s**2*(meannosys[k*nbins:(k+1)*nbins]-values[j])/err[j],color='k', label='realistic')
                ax[J,K].fill_between(s,s**2*((meannosys-errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meannosys+errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color='k',alpha=0.2)
                
                # sys with different weights
                covfitswp = '{}sys_FKP/EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1])
                hdu = fits.open(covfitswp)
                mocks = hdu[1].data['NGC+SGCmocks']
                Nmock = mocks.shape[1] 
                hdu.close()
                errbarsys = np.std(mocks,axis=1)/np.sqrt(Nmock)
                meansys = np.mean(mocks,axis=1)
                ax[J,K].plot(s,s**2*(meansys[k*nbins:(k+1)*nbins]-values[j])/err[j],color=colors[M], label='complete')
                ax[J,K].fill_between(s,s**2*((meansys-errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meansys+errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color=colors[M],alpha=0.2)
                if J==3:
                    plt.xlabel('s ($h^{-1}$Mpc)',fontsize=fontsize)
                    plt.xlim(-5,95)
                else:
                    plt.xticks([])
                if (j==0):
                    ax[J,K].set_ylabel('$s^2 * \\xi_{}$'.format(k*2),fontsize=fontsize)#('\\xi_{}$'.format(k*2))#
                    if k==0:
                        plt.ylim(-5,150)
                        plt.legend(loc=1)
                    else:
                        #plt.legend(loc=1)
                        plt.ylim(-90,30)
                if (j==1):
                    ax[J,K].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2),fontsize=fontsize)
                    if k==0:
                        plt.ylim(-5,75)
                        plt.yticks([0,50])
                    else:
                        plt.ylim(-10,10)
    elif function == 'wp':
        rscale = 'log'
        nbins  = 8
        nmu    = 80
        func   = 'wp'
        pimaxs  = [20,40,60,70,80]
        pimin = 5 #0
        swp = np.loadtxt(home+'binfile_log.dat',usecols=2)
        s = swp[(swp>binmin)&(swp<binmax+1)]
        K=1
        M=3
        # plot the wp multipoles   
        for k,pimax in enumerate([80,20]):
            wptailpi = 'wp_pi{}.fits.gz'.format(pimax)
            hdu = fits.open('{}nosys_FKP/EZmocks_{}_{}_{}_z{}z{}_{}'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1],wptailpi)) #
            mocks = hdu[1].data['NGC+SGCmocks']
            Nmock = mocks.shape[1] 
            hdu.close()
            errbarnosys = np.std(mocks,axis=1)/np.sqrt(Nmock)
            meannosys = np.mean(mocks,axis=1)

            # plot wp with different pi
            values=[np.zeros(nbins),meannosys]        
            err   = [np.ones(nbins),errbarnosys]
            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                ax[J,K].plot(s,(meannosys-values[j])/err[j],color='k', label='realistic')
                ax[J,K].fill_between(s,((meannosys-errbarnosys)-values[j])/err[j],((meannosys+errbarnosys)-values[j])/err[j],color='k',alpha=0.2)
                covfitswps  = '{}sys_FKP/EZmocks_{}_{}_{}_z{}z{}_{}'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1],wptailpi)
                hdu = fits.open(covfitswps) 
                mocks = hdu[1].data['NGC+SGCmocks']
                Nmock = mocks.shape[1] 
                hdu.close()
                errbarsys = np.std(mocks,axis=1)/np.sqrt(Nmock)
                meansys = np.mean(mocks,axis=1)
                ax[J,K].plot(s,(meansys-values[j])/err[j],color=colors[M], label='complete')
                ax[J,K].fill_between(s,((meansys-errbarsys)-values[j])/err[j],((meansys+errbarsys)-values[j])/err[j],color=colors[M],alpha=0.2)

                plt.xlabel('$r_p$ ($h^{-1}$Mpc)',fontsize=fontsize)
                if rscale=='log':
                    plt.xscale('log')

                if (j==0):
                    ax[J,K].set_ylabel('$w_p$',fontsize=fontsize)#('\\xi_{}$'.format(k*2))#
                    ax[J,K].text(5.5,8,'$\pi_{{max}} = {} h^{{-1}}$Mpc'.format(pimax))
                    plt.ylim(5.5,55)
                    plt.yscale('log')
                    if k==0:
                        plt.legend(loc=1)
                if (j==1):
                    ax[J,K].set_ylabel('$\Delta$ wp/err',fontsize=fontsize)
                    plt.ylim(-5,75)
                    plt.yticks([-5,50])

                # adjust labels
                from matplotlib.ticker import ScalarFormatter, NullFormatter
                for AX,axis in enumerate([ax[J,K].xaxis,ax[J,K].yaxis]):
                    axis.set_major_formatter(ScalarFormatter())
                    axis.set_minor_formatter(NullFormatter())
                    if (AX == 0)&(j==1):
                        ax[J,K].set_xticks([5,10,25])
                    elif (AX==1)&(j==0):
                        ax[J,K].set_yticks([10,20,40])
plt.savefig('wptest_sys-vs-nosys{}.png'.format(mockweight))
plt.close()


