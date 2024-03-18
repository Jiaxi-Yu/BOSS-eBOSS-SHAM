import numpy as np
from astropy.table import Table, vstack
import fitsio
import pylab as plt
import matplotlib.gridspec as gridspec
import sys
import os

c_kms = 299792.
min_dchi2 = 9
fc_limit = (62/3600)**2

# load the total catalogue and the repetitive samples
scratch  = os.environ['SCRATCH']+'/SHAM/catalog/SDSS_data/'
home     = os.environ['HOME']+'/SDSS_redshift_uncertainty/Vsmear-reproduce/'
gal      = sys.argv[1] #LOWZ, CMASS, LRG(not ready yet)
q_repeats= ['zerr','FIBER2FLUX','z']
q_clusts = ['ZERR_REDROCK','FIBER2FLUX','Z']
xlabels  = ['ZERR  (km/s)','i magnitude', 'redshift']

if gal == 'LRG':
    proj='eBOSS'
    zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
    maxdvs = [235,275,275,300,255,360]
    ylim = 100;xlim = 200
    #zmin = 0.6; zmax = 1.0
    clusteringname = scratch+'eBOSS_LRG_clustering_data-{}-vDR16.fits'
    caps = ['NGC','SGC']
elif gal == 'CMASS':
    proj='BOSS'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
    maxdvs = [205,200,235,270]
    ylim = 50;xlim = 100
    #zmin = 0.43; zmax = 0.7
    clusteringname = scratch+'galaxy_DR12v5_CMASS_{}.fits.gz'
    caps = ['North','South']
    magmin = 19.5
    magmax = 22
elif gal == 'LOWZ':
    proj='BOSS'
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
    maxdvs = [105,140,140]
    ylim = 40;xlim = 100
    #zmin = 0.2; zmax = 0.43
    clusteringname = scratch+'galaxy_DR12v5_LOWZ_{}.fits.gz'
    caps = ['North','South']
    magmin = 17.5
    magmax = 21.5
else:
    print("Wrong input")    

nrow = len(q_repeats)
fig = plt.figure(figsize=(4*len(zmins),4*nrow))
spec = gridspec.GridSpec(nrows=nrow,ncols=len(zmins),wspace=0.,hspace=0.25,left=0.08,right=0.99,top=0.95,bottom=0.08)
ax = np.empty((nrow,len(zmins)), dtype=type(plt.axes))
for k,zmin,zmax,maxdv in zip(range(len(zmins)),zmins,zmaxs,maxdvs):
    # repeat samples
    reobs = Table(fitsio.read(home+'clustering_zerr/{}_targetid_deltav_zerr.fits.gz'.format(gal)))
    print('the repetitive sample sample reading finished.')

    # clustering samples
    data_cap = []
    for cap in caps:            
        data_cap.append(Table(fitsio.read(clusteringname.format(cap))))
    clustering = vstack(data_cap)

    # binning the deltav
    binwidth = 5
    # plot the histogram from repeat and clustering
    for j in range(nrow):
        
        ax[j,k] = fig.add_subplot(spec[j,k])
        if j==0:
            bins = np.arange(0,maxdv+1,5)
            clustering[q_clusts[j]] = clustering[q_clusts[j]]*c_kms/clustering['Z']
        elif j==1:
            bins = np.arange(magmin,magmax,0.1)
            reobs[q_repeats[j]]     = 22.5 - 2.5 * np.log10(reobs[q_repeats[j]][:,3])
            clustering[q_clusts[j]] = 22.5 - 2.5 * np.log10(clustering[q_clusts[j]][:,3])
        elif j==2:
            bins = np.arange(zmin,zmax+0.01,0.01)
        ax[j,k].hist(reobs[q_repeats[j]],bins = bins, color='b',label='clustring repeat',density=True, histtype='step')
        ax[j,k].hist(clustering[q_clusts[j]],bins=bins,color='r',label='clustering all',alpha=0.3,density=True)
        ax[j,k].set_xlabel(xlabels[j])
        plt.title('{} at {}<z<{}'.format(gal,zmin,zmax))
        if k==0:
            ax[j,k].set_ylabel('normalised counts')
            plt.legend(loc=2)
        else:
            plt.yticks(alpha=0)
plt.savefig(home+'clustering_zerr/{}_dv-representative.png'.format(gal))
plt.close()


