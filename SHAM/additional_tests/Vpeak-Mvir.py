import matplotlib 
matplotlib.use('agg')
import time
time_start=time.time()
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from astropy.io import fits
import matplotlib.gridspec as gridspec
import os
import sys

# data reading
path =  '/global/cscratch1/sd/jiaxi/master/catalog/'
MultiDark = fits.open(path+'MD_hlist_0.53780-subhalo.fits.gz')
UNIT      = fits.open(path+'UNIT_hlist_0.53780-subhalo.fits.gz')
# halo(PID==-1) or subhalo(PID!=-1)
MDP = MultiDark[1].data['PID']
UNIP =UNIT[1].data['PID']
# Mvir-Vpeak
fig = plt.figure(figsize = (12, 5)) 
spec = gridspec.GridSpec(ncols=2, nrows=1,  hspace=0.3,wspace=0.5)
ax = np.empty((1,2), dtype=type(plt.axes))

for i,function in enumerate(['halo','subhalo']):
    ax[0,i] = fig.add_subplot(spec[0,i])
    if function == 'halo':
        ax[0,i].scatter(MultiDark[1].data['Mvir'][MDP==-1],MultiDark[1].data['Vpeak'][MDP==-1]**2,c='r',alpha=0.4,s=1,label = 'MultiDark')
        ax[0,i].scatter(UNIT[1].data['Mvir'][UNIP==-1],UNIT[1].data['Vpeak'][UNIP==-1]**2,c='b',alpha=0.4,s=1,label = 'UNIT')
    if function =='subhalo':
        ax[0,i].scatter(MultiDark[1].data['Mvir'][MDP!=-1],MultiDark[1].data['Vpeak'][MDP!=-1]**2,c='r',alpha=0.4,s=1,label = 'MultiDark')
        ax[0,i].scatter(UNIT[1].data['Mvir'][UNIP!=-1],UNIT[1].data['Vpeak'][UNIP!=-1]**2,c='b',alpha=0.4,s=1,label = 'UNIT')
        
    plt.xlabel('Mvir ($M_{sun}/h$)')
    plt.ylabel('$V_{peak}^2 (km/s)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e8,5e14)
#plt.ylim(2.5e13,1.2*10**15)
#plt.plot(np.logspace(13,14.7,6),np.logspace(13,14.7,6),'k--',label='$M_{sub}=M_{parent}$')
plt.legend(loc=0)
plt.title('Vpeak - Mvir relation for subhaloes')
plt.savefig('Vpeak - Mvir.png')
plt.close('all')
