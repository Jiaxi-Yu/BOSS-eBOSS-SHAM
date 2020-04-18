import matplotlib 
matplotlib.use('agg')
import time
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from astropy.io import fits
import matplotlib.gridspec as gridspec
import os

path =  '/global/cscratch1/sd/jiaxi/master/catalog/'
MultiDark = fits.open(path+'MD_hlist_0.53780.fits.gz')
UNIT      = fits.open(path+'UNIT_hlist_0.53780.fits.gz')
MD = MultiDark[1].data['Mvir']
UNI =UNIT[1].data['Mvir']

# Mvir distribution
Mbins=np.logspace(np.log10(10**9.3),np.log10(10**15.1),50+1) 
num1 = np.histogram(UNI,bins=Mbins)#,alpha=0.5,label='UNIT',color='r')
num2 = np.histogram(MD,bins=Mbins)#,alpha=0.5,label='MultiDark',color='b')
# store the data
file = 'Mvir_distribution.txt'
ascii.write(Table([Mbins[:-1],Mbins[1:],num1[0],num2[0]],names=('M_min', 'M_max', 'MD', 'UNIT')),file,delimiter='\t',overwrite=True)


# vpeak distribution function
Vbins=np.linspace(20,1500,50+1) 
num11 = np.histogram(UNIT[1].data['Vpeak'],bins=Vbins)#,alpha=0.5,label='UNIT',color='r')
num21 = np.histogram(MultiDark[1].data['Vpeak'],bins=Vbins)#,alpha=0.5,label='MultiDark',color='b')
# store the data
file2 = 'vmax_distribution.txt'
ascii.write(Table([Vbins[:-1],Vbins[1:],num11[0],num21[0]],names=('V_min', 'V_max', 'MD', 'UNIT')),file2,delimiter='\t',overwrite=True)


# the vpeak-mvir relation(inappropriate)
'''
mean,median = np.zeros((50,2)),np.zeros((50,2)) 
for i in range(50): 
    median[i,0]=np.nanmedian(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    median[i,1]=np.nanmedian(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])]) 
    mean[i,0]=np.nanmean(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    mean[i,1]=np.nanmean(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])])
# plot the vpeak-mvir relation(inappropriate)
fig = plt.figure(figsize = (7, 6)) 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.ones(50),median[:,0]]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].plot(bins[:-1],median[:,0]/value,c='r',label = 'MD_median')
    ax[k,0].plot(bins[:-1],median[:,1]/value,c='b',label = 'UNIT_median')
        

    ax[k,0].set_xlabel('$M_{vir}$')
    if k==0:
        ax[k,0].set_ylim(0,2000)
        ax[k,0].set_ylabel('stacked $V_{peak}$') 
        plt.xscale('log');
        plt.legend(loc=0) 
    else:
        ax[k,0].set_ylim(-100,100)
        ax[k,0].set_ylabel('$V_{peak}$ diff')
        plt.xscale('log');
            
plt.savefig(path[:-8]+'Vmax-Mvir_cmoparison_median_test.png')
plt.close()
'''

# Mvir distribution plot
fig = plt.figure(figsize = (7, 6)) 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.ones(50),num2[0]]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].step(bins[:-1],num1[0]/value,c='r',label = 'MultiDark',alpha=0.5)
    ax[k,0].step(bins[:-1],num2[0]/value,c='b',label = 'UNIT',alpha=0.5)
        
    ax[k,0].set_xlabel('$M_{vir}$')
    if k==0:
        plt.title('the Mvir distribution comparison') 
        ## title cannot be set for the whole figure(outside of the loop) 
        ## otherwise, subplots will stick together.
        ax[k,0].set_ylim(1e0,1e8)
        ax[k,0].set_ylabel('# of galoes') 
        plt.xscale('log');
        plt.yscale('log')
        plt.legend(loc=0) 
    else:
        ax[k,0].set_ylim(1e-1,10)
        ax[k,0].set_ylabel('halo # ratio')
        plt.xscale('log');
        plt.yscale('log')
            
plt.savefig(path[:-8]+'Mvir_distribution.png')
plt.close()

# vmax distribution plot
fig = plt.figure(figsize = (7, 6)) 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.ones(50),num21[0]]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].step(Vbins[:-1],num11[0]/value,c='r',label = 'MultiDark',alpha=0.5)
    ax[k,0].step(Vbins[:-1],num21[0]/value,c='b',label = 'UNIT',alpha=0.5)
        
    ax[k,0].set_xlabel('$M_{vir}$')
    if k==0:
        plt.title('the Vmax distribution comparison')
        ax[k,0].set_ylim(1e0,3e8)
        ax[k,0].set_ylabel('# of galoes') 
        plt.yscale('log')
        plt.legend(loc=0) 
    else:
        ax[k,0].set_ylim(1e-1,10)
        ax[k,0].set_ylabel('halo # ratio')
        plt.yscale('log')
            
plt.savefig(path[:-8]+'Vmax_distribution.png')
plt.close()