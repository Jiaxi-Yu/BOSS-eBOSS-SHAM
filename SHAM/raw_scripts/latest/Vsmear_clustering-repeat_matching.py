import numpy as np
from astropy.io import fits
from multiprocessing import Pool 
from itertools import repeat
import pylab as plt
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
    #zmin = 0.6; zmax = 1.0
    clusteringN,clusteringS = scratch+'eBOSS_clustering_fits/eBOSS_LRG_clustering_NGC_v7_2.dat.fits',scratch+'eBOSS_clustering_fits/eBOSS_LRG_clustering_SGC_v7_2.dat.fits'
elif gal == 'CMASS':
    proj='BOSS'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
    maxdvs = [205,200,235,270]
    #zmin = 0.43; zmax = 0.7
    clusteringN,clusteringS = scratch+'BOSS_data/galaxy_DR12v5_CMASS_North.fits.gz',scratch+'BOSS_data/galaxy_DR12v5_CMASS_South.fits.gz'
elif gal == 'LOWZ':
    proj='BOSS'
    zmins = [0.2]#[0.2, 0.33,0.2]
    zmaxs = [0.43]#[0.33,0.43,0.43]
    maxdvs = [140]#[105,140,140]
    #zmin = 0.2; zmax = 0.43
    clusteringN,clusteringS = scratch+'BOSS_data/galaxy_DR12v5_LOWZ_North.fits.gz',scratch+'BOSS_data/galaxy_DR12v5_LOWZ_South.fits.gz'
else:
    print("Wrong input")    

fig = plt.figure(figsize=(7*len(zmin),6))
spec = gridspec.GridSpec(nrows=1,ncols=3,wspace=0.4)
ax = np.empty((1,3), dtype=type(plt.axes))
k  = 0
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    # repeat samples
    repeatfile = home+'{}-{}_deltav_z{}z{}.fits.gz'.format(proj,gal,zmin,zmax)
    hdu = fits.open(repeatfile)
    reobs = hdu[1].data
    reobs = reobs[reobs['delta_chi2']>min_dchi2]
    hdu.close()
    print('the repetitive sample sample reading finished.')

    # clustering matched deltav and zerr
    hdu = fits.open(home+'clustering_zerr1/{}_targetid_deltav_zerr_z{}z{}.fits.gz'.format(gal,zmin,zmax))
    data = hdu[1].data
    hdu.close()
    LRGtot = np.zeros((len(data['zerr']),3))
    for k,name in enumerate(data.columns.names):
        LRGtot[:,k] = data[name]
    
    # binning the deltav
    binwidth = 5
    bins = np.arange(-max_dv, max_dv+1, binwidth)
    dens,BINS = np.histogram(dvsel,bins=bins)
    norm = np.sum(dens)
    dens = dens/norm

    # plot the histogram from repeat and clustering
    ax[0,k] = fig.add_subplot(spec[0,k])
    plt.scatter()
    
    k+=1

