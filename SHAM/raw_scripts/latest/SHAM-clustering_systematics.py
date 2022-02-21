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
        # no sys data
        hdu = fits.open('{}nosys_FKP/EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1])) #
        mocks = hdu[1].data['NGC+SGCmocks']
        Nmock = mocks.shape[1] 
        hdu.close()
        errbarnosys = np.std(mocks,axis=1)
        meannosys = np.mean(mocks,axis=1)
        # sys data
        covfitswp = '{}sys_FKP/EZmocks_{}_{}_{}_z{}z{}_quad.fits.gz'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1])
        hdu = fits.open(covfitswp)
        mocks = hdu[1].data['NGC+SGCmocks']
        Nmock = mocks.shape[1] 
        hdu.close()
        errbarsys = np.std(mocks,axis=1)
        meansys = np.mean(mocks,axis=1)
        mockweight = '_FKPSYSTOT'
        for name,k in zip(['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),meannosys[k*nbins:(k+1)*nbins]]        
            err   = [np.ones(nbins),s**2*np.sqrt(errbarnosys[k*nbins:(k+1)*nbins]**2+errbarsys[k*nbins:(k+1)*nbins]**2)]
            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                # nosys: black & grey shade
                ax[J,K].plot(s,s**2*(meannosys[k*nbins:(k+1)*nbins]-values[j])/err[j],color='k', label='w/o syst.')
                ax[J,K].fill_between(s,s**2*((meannosys-errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meannosys+errbarnosys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color='k',alpha=0.2)
                # sys: coloured shade                
                ax[J,K].plot(s,s**2*(meansys[k*nbins:(k+1)*nbins]-values[j])/err[j],color=colors[M], label='with syst.')
                ax[J,K].fill_between(s,s**2*((meansys-errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],s**2*((meansys+errbarsys)[k*nbins:(k+1)*nbins]-values[j])/err[j],color=colors[M],alpha=0.2)
                if J==3:
                    plt.xlabel('$s\,(h^{-1}\,Mpc)$',fontsize=fontsize)
                    plt.xlim(-5,95)
                else:
                    plt.xticks([])
                if (j==0):
                    ax[J,K].set_ylabel('$s^2 \\xi_{}\,(h^{{-2}}\,Mpc^2)$'.format(k*2),fontsize=fontsize)#('\\xi_{}$'.format(k*2))#
                    if k==0:
                        plt.ylim(-10,155)
                        plt.legend(loc=1,fontsize=fontsize-2)
                    else:
                        plt.ylim(-110,30)
                if (j==1):
                    if k==0:
                        ax[J,K].set_ylabel(r'$\Delta\xi_0/\bar{\epsilon}_{{obs,\xi_0}}$',fontsize=fontsize)
                    else:
                        ax[J,K].set_ylabel(r'$\Delta\xi_2/\bar{\epsilon}_{{obs,\xi_2}}$',fontsize=fontsize)                    
                    plt.ylim(-3.5,3.5)
                    plt.yticks([-3,0,3])
                    
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
            # no sys
            hdu = fits.open('{}nosys_FKP/EZmocks_{}_{}_{}_z{}z{}_{}'.format(datapath,'nosys',function,rscale,Zrange[0],Zrange[1],wptailpi)) #
            mocks = hdu[1].data['NGC+SGCmocks']
            Nmock = mocks.shape[1] 
            hdu.close()
            errbarnosys = np.std(mocks,axis=1)
            meannosys = np.mean(mocks,axis=1)
            # sys
            covfitswps  = '{}sys_FKP/EZmocks_{}_{}_{}_z{}z{}_{}'.format(datapath,'sys',function,rscale,Zrange[0],Zrange[1],wptailpi)
            hdu = fits.open(covfitswps) 
            mocks = hdu[1].data['NGC+SGCmocks']
            Nmock = mocks.shape[1] 
            hdu.close()
            errbarsys = np.std(mocks,axis=1)
            meansys = np.mean(mocks,axis=1)
            # plot wp with different pi
            values=[np.zeros(nbins),meannosys]        
            err   = [np.ones(nbins),np.sqrt(errbarnosys**2+errbarsys**2)]
            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                ax[J,K].plot(s,(meannosys-values[j])/err[j],color='k', label='w/o syst.')
                ax[J,K].fill_between(s,((meannosys-errbarnosys)-values[j])/err[j],((meannosys+errbarnosys)-values[j])/err[j],color='k',alpha=0.2)
                ax[J,K].plot(s,(meansys-values[j])/err[j],color=colors[M], label='with syst.')
                ax[J,K].fill_between(s,((meansys-errbarsys)-values[j])/err[j],((meansys+errbarsys)-values[j])/err[j],color=colors[M],alpha=0.2)

                plt.xlabel(r'$r_p\,(h^{-1}\,Mpc$)',fontsize=fontsize)
                if rscale=='log':
                    plt.xscale('log')

                #plt.yscale('log')
                if (j==0):
                    ax[J,K].set_ylabel('$w_p$',fontsize=fontsize)#('\\xi_{}$'.format(k*2))#
                    ax[J,K].text(5.5,8,r'$\pi_{{max}} = {}\,h^{{-1}}\,Mpc$'.format(pimax))
                    plt.ylim(5.5,44)
                    if k==0:
                        plt.legend(loc=1,fontsize=fontsize-2)
                if (j==1):
                    ax[J,K].set_ylabel(r'$\Delta w_p/\bar{\epsilon}_{{obs,w_p}}$',fontsize=fontsize)
                    plt.ylim(-2.9,2.9)
                    plt.yticks([-2,0,2])

                # adjust labels
                from matplotlib.ticker import ScalarFormatter, NullFormatter
                for AX,axis in enumerate([ax[J,K].xaxis,ax[J,K].yaxis]):
                    axis.set_major_formatter(ScalarFormatter())
                    axis.set_minor_formatter(NullFormatter())
                    if (AX == 0)&(j==1):
                        ax[J,K].set_xticks([5,10,25])
                    elif (AX==1)&(j==0):
                        ax[J,K].set_yticks([10,20,40])
plt.savefig('wptest_sys-vs-nosys{}.pdf'.format(mockweight))
plt.close()


