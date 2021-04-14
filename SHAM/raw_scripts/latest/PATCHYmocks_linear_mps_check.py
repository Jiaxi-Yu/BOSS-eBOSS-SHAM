#!/usr/bin/env python3 
import matplotlib  
matplotlib.use('agg') 
import numpy as np 
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack 
import matplotlib.pyplot as plt 
import glob 
import matplotlib.gridspec as gridspec 
import sys 

zrange = sys.argv[1]#'z0.57z0.7'
if sys.argv[3] =='S':
    cap='South'
else:
    cap='North'
mocks = glob.glob('/hpcstorage/jiayu/PATCHY/'+sys.argv[2]+'_1200/'+zrange+'/2PCF/2PCF_PATCHYmock_'+sys.argv[2]+'_'+sys.argv[3]+'GC_DR12_'+zrange+'_*.xi') 

# part
SMIN=5;SMAX=25
obscut = np.loadtxt('/hpcstorage/jiayu/BOSS_clustering/2PCF/OBS_'+sys.argv[2]+'_'+cap+'_DR12v5_'+zrange+'.mps')
sel = (obscut[:,0]>=SMIN)&(obscut[:,0]<=SMAX)
s = obscut[:,0][sel]
obscut = obscut[sel,3:]
mpscut = [i for i in range(len(mocks))]
for m,mock in enumerate(mocks):    
    print(mock)
    mpscut[m] = np.loadtxt(mock[:-2]+'mps')[sel,3:]
mpsmean =  np.mean(mpscut,axis=0)
mpsstd = np.std(mpscut,axis=0)

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

plt.savefig('{}_s5s25_{}_{}.png'.format(sys.argv[2],zrange,cap),bbox_tight=True)
plt.close()
"""
def cf(xifile,isobs=True):
    ds=5;ns=100;nmu=120
    smin,smax,mumin,mumax,mono = np.loadtxt(xifile,unpack=True)
    mu = (mumin+mumax)/2
    mask = np.isnan(mono)
    mono[mask] = 0
    sbin = int(ns / ds)
    se = np.linspace(smin[0], smax[-1], sbin+1)
    s = (se[1:] + se[:-1]) * 0.5

    ######################################################
    # wrong dimension reshape
    mu = np.median(mu.reshape([sbin, ds, nmu]), axis=1)
    cnt = np.sum(mono.reshape([sbin, ds, nmu]), axis=1)
    quad = cnt * 2.5 * (3 * mu**2 - 1)
    hexa = cnt * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    return [s,np.sum(cnt,axis=1)/nmu,np.sum(quad,axis=1)/nmu,np.sum(hexa,axis=1)/nmu]

    ######################################################

obs  = cf('/hpcstorage/jiayu/BOSS_clustering//2PCF/OBS_'+sys.argv[2]+'_'+cap+'_DR12v5_'+zrange+'.xi')

# full
mps = [i for i in range(len(mocks))]
for j,mock in enumerate(mocks):
    mps[j] = cf(mock[:-2]+'{}',isobs=False)

fig = plt.figure(figsize=(21,8))
spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
ax = np.empty((2,3), dtype=type(plt.axes))
for k in range(3):
#for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
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

plt.savefig('{}_s{}s{}_{}_{}.png'.format(sys.argv[2],0,100,zrange,cap),bbox_tight=True)
plt.close()

"""