#!/usr/bin/env python3 
import matplotlib  
matplotlib.use('agg') 
import numpy as np 
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack 
import matplotlib.pyplot as plt 
from multiprocessing import Pool
import glob 
from astropy.io import fits
import matplotlib.gridspec as gridspec 
import sys 
import os

zrange = sys.argv[1]#'z0.57z0.7'
gal = sys.argv[2]
if sys.argv[3] =='SGC':
    cap='South'
elif sys.argv[3] =='NGC':
    cap='North'
else:
    cap = 'NGC+SGC'
mocks = glob.glob('/hpcstorage/jiayu/PATCHY/{}/{}/2PCF/2PCF_PATCHYmock_{}_{}_DR12_{}_*.xi'.format(gal,zrange,gal,cap,zrange)) 

# part
SMIN=5;SMAX=25
home = '/home/astro/jiayu/Desktop/SHAM/catalog/'
obs = np.loadtxt('{}BOSS_zbins_mps/OBS_{}_{}_DR12v5_{}.mps'.format(home,gal,cap,zrange))
sel = (obs[:,0]>=SMIN)&(obs[:,0]<=SMAX)
s = obs[:,0][sel]
obscut = obs[sel,3:]
covfits = '{}BOSS_zbins_mps/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,'linear',zrange,'quad')
if os.path.exists(covfits):
    hdu = fits.open(covfits) #
    mock = hdu[1].data[cap+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    Nstot=100
    mocks = vstack((mock[SMIN:SMAX,:],mock[SMIN+Nstot:SMAX+Nstot,:]))
    mpsmean =  np.mean(mocks,axis=1)
    mpsstd = np.std(mocks,axis=1)
    #import pdb;pdb.set_trace()
    # plot 5-25Mpc/h for SHAM
    fig = plt.figure(figsize=(12,7))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for k in range(2):
        values=[np.zeros_like(s),obscut[:,k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscut[:,k]-values[j]),s**2*mpsstd[k*len(s):(k+1)*len(s)], marker='^',ecolor='k',ls="none",label='obs')
            ax[j,k].fill_between(s,s**2*((mpsmean-mpsstd)[k*len(s):(k+1)*len(s)]-values[j]),s**2*((mpsmean+mpsstd)[k*len(s):(k+1)*len(s)]-values[j]),label='PATHCYmocks 1$\sigma$',color='c')
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('s^2 * $\\xi_{}$'.format(k*2))
                plt.legend(loc=0)
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig('{}_s5s25_{}_{}.png'.format(gal,zrange,cap),bbox_tight=True)
    plt.close()
else:
    # load file in threads
    def loadmps(name):
        return np.loadtxt(name[:-2]+'mps')[sel,3:]

    # s binwidth 5Mpc/h function
    def cf(xifile):
        ds=5;ns=100;nmu=120
        smin,smax,mumin,mumax,mono = np.loadtxt(xifile,unpack=True)
        mu = (mumin+mumax)/2
        mask = np.isnan(mono)
        mono[mask] = 0
        sbin = int(ns / ds)
        se = np.linspace(smin[0], smax[-1], sbin+1)
        s = (se[1:] + se[:-1]) * 0.5
        ######################################################
        # reshape style different from EZmocks
        mu = np.median(mu.reshape([nmu,sbin, ds]), axis=-1)
        cnt = np.sum(mono.reshape([nmu,sbin, ds]), axis=-1)
        quad = cnt * 2.5 * (3 * mu**2 - 1)
        hexa = cnt * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        return [s,np.sum(cnt,axis=0)/nmu/ds,np.sum(quad,axis=0)/nmu/ds,np.sum(hexa,axis=0)/nmu/ds]

        ######################################################


    mpscut = [i for i in range(len(mocks))]
    mps = [i for i in range(len(mocks))]
    pool = Pool()    
    for n, temp_array in enumerate(pool.imap(loadmps,mocks)):
        mpscut[n]= temp_array 
    for n, temp in enumerate(pool.imap(cf,mocks)):
        mps[n]= temp
    pool.close() 
    pool.join()
    mpsmean =  np.mean(mpscut,axis=0)
    mpsstd = np.std(mpscut,axis=0)

    # plot 5-25Mpc/h for SHAM
    fig = plt.figure(figsize=(21,8))
    spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,3), dtype=type(plt.axes))
    for k in range(3):
        values=[np.zeros_like(s),obscut[:,k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscut[:,k]-values[j]),s**2*mpsstd[:,k], marker='^',ecolor='k',ls="none",label='obs')
            ax[j,k].fill_between(s,s**2*((mpsmean-mpsstd)[:,k]-values[j]),s**2*((mpsmean+mpsstd)[:,k]-values[j]),label='PATHCYmocks 1$\sigma$',color='c')
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('s^2 * $\\xi_{}$'.format(k*2))
                plt.legend(loc=0)
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig('{}_s5s25_{}_{}.png'.format(gal,zrange,cap),bbox_tight=True)
    plt.close()


    # s in (0,100]
    # plot
    fig = plt.figure(figsize=(21,8))
    spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,3), dtype=type(plt.axes))
    for k in range(3):
        values=[np.zeros_like(obs[0]),obs[k+1]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(obs[0],obs[0]**2*(obs[k+1]-values[j]),obs[0]**2*(np.std(mps,axis=0)[k+1]), marker='^',ecolor='k',ls="none",label='obs')
            ax[j,k].fill_between(obs[0],obs[0]**2*((np.mean(mps,axis=0)-np.std(mps,axis=0))[k+1]-values[j]),obs[0]**2*((np.mean(mps,axis=0)+np.std(mps,axis=0))[k+1]-values[j]),label='PATHCYmocks 1$\sigma$',color='c')
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('s^2 * $\\xi_{}$'.format(k*2))
                plt.legend(loc=0)
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig('{}_s{}s{}_{}_{}.png'.format(gal,0,100,zrange,cap),bbox_tight=True)
    plt.close()

