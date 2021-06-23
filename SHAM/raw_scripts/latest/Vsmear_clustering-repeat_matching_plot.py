import numpy as np
from astropy.io import fits
from multiprocessing import Pool 
from itertools import repeat
import pylab as plt
import matplotlib.gridspec as gridspec
import sys
import os

c_kms = 299792.
min_dchi2 = 9
fc_limit = (62/3600)**2

# load the total catalogue and the repetitive samples
scratch = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
home = '/global/homes/j/jiaxi/Vsmear/'
gal = sys.argv[1]
if gal == 'LRG':
    proj='eBOSS'
    zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
    maxdvs = [235,275,275,300,255,360]
    ylim = 100;xlim = 200
    #zmin = 0.6; zmax = 1.0
    clusteringN,clusteringS = scratch+'eBOSS_clustering_fits/eBOSS_LRG_clustering_NGC_v7_2.dat.fits',scratch+'eBOSS_clustering_fits/eBOSS_LRG_clustering_SGC_v7_2.dat.fits'
elif gal == 'CMASS':
    proj='BOSS'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
    maxdvs = [205,200,235,270]
    ylim = 50;xlim = 100
    #zmin = 0.43; zmax = 0.7
    clusteringN,clusteringS = scratch+'BOSS_data/galaxy_DR12v5_CMASS_North.fits.gz',scratch+'BOSS_data/galaxy_DR12v5_CMASS_South.fits.gz'
elif gal == 'LOWZ':
    proj='BOSS'
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
    maxdvs = [105,140,140]
    ylim = 40;xlim = 100
    #zmin = 0.2; zmax = 0.43
    clusteringN,clusteringS = scratch+'BOSS_data/galaxy_DR12v5_LOWZ_North.fits.gz',scratch+'BOSS_data/galaxy_DR12v5_LOWZ_South.fits.gz'
else:
    print("Wrong input")    

nrow = 3
fig = plt.figure(figsize=(7*len(zmins),21))
spec = gridspec.GridSpec(nrows=nrow,ncols=len(zmins),wspace=0.3,hspace=0.3)
ax = np.empty((nrow,len(zmins)), dtype=type(plt.axes))
for k,zmin,zmax,maxdv in zip(range(len(zmins)),zmins,zmaxs,maxdvs):
    # repeat samples
    repeatfile = home+'{}-{}_deltav_z{}z{}.fits.gz'.format(proj,gal,zmin,zmax)
    hdu = fits.open(repeatfile)
    reobs = hdu[1].data
    reobs = reobs[(reobs['delta_chi2']>min_dchi2)&(abs(reobs['delta_v'])<1000)]
    hdu.close()
    print('the repetitive sample sample reading finished.')

    # clustering matched deltav and zerr
    LRG = []
    for cap in ['N','S']:
        hdu = fits.open(home+'clustering_zerr/{}_targetid_deltav_zerr_z{}z{}-{}.fits.gz'.format(gal,zmin,zmax,cap))
        data = hdu[1].data
        hdu.close()
        LRGtot = np.zeros((len(data['zerr']),3))
        for n,name in enumerate(data.columns.names):
            LRGtot[:,n] = data[name]
        LRG.append(LRGtot)
    clustering = np.vstack((LRG[0],LRG[0]))

    # binning the deltav
    binwidth = 5
    bins = np.arange(-maxdv, maxdv+1, binwidth)
    dens,BINS = np.histogram(reobs['delta_v'],bins=bins)
    norm = np.sum(dens)
    dens = dens/norm
    dens1,BINS = np.histogram(clustering[:,1][~np.isnan(clustering[:,1])],bins=bins)
    norm1 = np.sum(dens1)
    dens1 = dens1/norm1
    binmid = (bins[1:]+bins[:-1])/2

    bins1 = np.arange(0, maxdv/2, 1)
    dens0,BINS = np.histogram(clustering[:,-1][~np.isnan(clustering[:,1])],bins=bins1)
    norm0 = np.sum(dens0)
    dens0 = dens0/norm0
    dens3,BINS = np.histogram(clustering[:,-1][~np.isnan(clustering[:,-1])],bins=bins1)
    norm3 = np.sum(dens3)
    dens3 = dens3/norm3
    binmid1 = (bins1[1:]+bins1[:-1])/2

    # plot the histogram from repeat and clustering
    for j in range(nrow):
        ax[j,k] = fig.add_subplot(spec[j,k])
        if j==0:
            ax[j,k].plot(binmid,dens,c='b',label='all repeat ')
            ax[j,k].plot(binmid,dens1,c='r',label='clustering repeat')
            plt.title('histogram: {}<z<{} all repeat $\Delta$v v.s. clustering repeat $\Delta$v'.format(zmin,zmax))
            ax[j,k].set_ylabel('normalised counts')
            ax[j,k].set_xlabel('$\Delta$v (km/s)')
        elif j==1:
            ax[j,k].plot(binmid1,dens0,c='b',label='clustering repeat')
            ax[j,k].plot(binmid1,dens3,c='r',label='clustering all')
            plt.title('histogram: {}<z<{} clustering repeat ZERR v.s. clustering all ZERR'.format(zmin,zmax))
            ax[j,k].set_ylabel('normalised counts')
            ax[j,k].set_xlabel('ZERR (km/s)')
        else:
            hb = ax[j,k].hexbin(reobs['delta_v'],reobs['zerr'],cmap='Blues')#,c='b',label='ZERRavg',alpha=0.5)
            #ax[j,k].scatter(reobs['delta_v'],reobs['zerr0'],c='m',label='ZERR0',alpha=0.5)
            #ax[j,k].scatter(reobs['delta_v'],reobs['zerr1'],c='green',label='ZERR1',alpha=0.5)
            # if no difference, can use ax[j,k].hexbin
            ax[j,k].set_ylabel('ZERR')
            ax[j,k].set_xlabel('$\Delta$v (km/s)')
            ax[j,k].set_ylim(0,ylim)
            ax[j,k].set_xlim(-xlim,xlim)
            plt.title('scatter: {}<z<{} repeat $\Delta$v v.s. repeat ZERR'.format(zmin,zmax))
            cb = fig.colorbar(hb, ax=ax[j,k])

        plt.legend(loc=1)
plt.savefig(home+'clustering_zerr/{}_dv-representative.png'.format(gal))
plt.close()

"""
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    # clustering matched deltav and zerr
    LRG = []
    for i,cap in enumerate(['N','S']):
        hdu = fits.open(home+'clustering_zerr/{}_targetid_deltav_zerr_z{}z{}-{}.fits.gz'.format(gal,zmin,zmax,cap))
        data = hdu[1].data
        hdu.close()
        LRGtot = np.zeros((len(data['zerr']),3))
        for k,name in enumerate(data.columns.names):
            LRGtot[:,k] = data[name]
        LRG.append(LRGtot)
    clustering = np.vstack((LRG[0],LRG[0]))

    # figure: dv vs zerr
    bins = np.arange(0,maxdv+1,5)
    densdv,BINS = np.histogram(np.abs(clustering[~np.isnan(clustering[:,1]),1]),bins=bins)
    denszerr,BINS = np.histogram(np.abs(clustering[~np.isnan(clustering[:,2]),2]),bins=bins)
    plt.plot((BINS[1:]+BINS[:-1])/2,densdv/sum(densdv),'r',label='$\Delta$ v')
    plt.plot((BINS[1:]+BINS[:-1])/2,denszerr/sum(denszerr),'b',label='ZERR')
    plt.yscale('log')
    plt.legend(loc=1)
    plt.title('{} clustering galaxy |$\Delta$ v| (if exists) v.s. ZERR'.format(gal))
    plt.savefig(home+'clustering_zerr/{}_clustering_deltav_vs_zerr_z{}z{}.png'.format(gal,zmin,zmax))
    plt.close()
"""