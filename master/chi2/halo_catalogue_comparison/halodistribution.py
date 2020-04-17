import matplotlib 
matplotlib.use('agg')
import time
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from astropy.io import fits

path =  '/global/cscratch1/sd/jiaxi/master/catalog/'
MultiDark = fits.open(path+'MD_hlist_0.53780.fits.gz')
UNIT      = fits.open(path+'UNIT_hlist_0.53780.fits.gz')
MD = MultiDark[1].data['Mvir']
UNI =UNIT[1].data['Mvir']

fig,ax = plt.subplots()
plt.hist(UNI,bins=50,alpha=0.5,label='UNIT',color='r')
plt.hist(MD,bins=50,alpha=0.5,label='MultiDark',color='b')
plt.title('halo catalogue vpeak distribution')
plt.legend(loc=0)
plt.xlabel('$M_{vir}$')
plt.ylabel('# of halos')
plt.yscale('log');
plt.xscale('log');
plt.savefig(path[:-8]+'halomass_comparison.png')
plt.close()

fig,ax = plt.subplots()
#plt.scatter(UNI,UNIT[1].data['Vpeak'],alpha=0.5,label='UNIT',color='r')
#plt.scatter(MD,MultiDark[1].data['Vpeak'],alpha=0.5,label='MultiDark',color='b')
mean,median = np.zeros((50,2)),np.zeros((50,2)) 
for i in range(50): 
    median[i,0]=np.nanmedian(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    median[i,1]=np.nanmedian(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])]) 
    mean[i,0]=np.nanmean(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    mean[i,1]=np.nanmean(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])])
#plt.plot(bins[:-1],mean[:,0],c='r',label = 'MD_mean')
plt.plot(bins[:-1],median[:,0],c='r',label = 'MD_median')
#plt.plot(bins[:-1],mean[:,1],c='b',label = 'UNIT_mean')
plt.plot(bins[:-1],median[:,1],c='b',label = 'UNIT_median')
plt.xlabel
plt.ylabel('stacked $V_{max}$')
plt.title('$V_{max}-M_{vir}$ relations comparison')
plt.legend(loc=0)


fig = plt.figure(figsize = (5, 6))
import matplotlib.gridspec as gridspec 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.zeros(50),median[:,0]]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].plot(bins[:-1],median[:,0]-value,c='r',label = 'MD_median')
    ax[k,0].plot(bins[:-1],median[:,1]-value,c='b',label = 'UNIT_median')
        

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
            
plt.savefig(path[:-8]+'Vmax-Mvir_cmoparison_median.png')
plt.close()