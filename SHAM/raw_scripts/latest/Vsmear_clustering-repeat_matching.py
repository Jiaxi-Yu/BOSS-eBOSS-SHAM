import numpy as np
from astropy.io import fits
from multiprocessing import Pool 
from itertools import repeat
import pylab as plt
import sys
import os

c_kms = 299792.
min_dchi2 = 9

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
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
    maxdvs = [105,140,140]
    #zmin = 0.2; zmax = 0.43
    clusteringN,clusteringS = scratch+'BOSS_data/galaxy_DR12v5_LOWZ_North.fits.gz',scratch+'BOSS_data/galaxy_DR12v5_LOWZ_South.fits.gz'
else:
    print("Wrong input")    


hdu = fits.open('/global/cscratch1/sd/jiaxi/SHAM/catalog/spAll-zbest-v5_13_0.fits')
complete = hdu[1].data
hdu.close()
targetid_tot = complete['PLATE']*1e9+complete['MJD']*1e4+complete['FIBERID']
print('the complete sample reading finished.')
    
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    repeatfile = home+'{}-{}_deltav_z{}z{}.fits.gz'.format(proj,gal,zmin,zmax)
    hdu = fits.open(repeatfile)
    reobs = hdu[1].data
    reobs = reobs[reobs['delta_chi2']>min_dchi2]
    hdu.close()
    print('the repetitive sample sample reading finished.')


    # the matching function
    nthreads = 64
    def matching(index,clustering):
        #print('thread ',index)
        #slice the clustering catalogue in nthread parts
        partlen = len(clustering)//nthreads
        if index == nthreads-1:
            TARGETID = clustering[index*partlen:]
        else:
            TARGETID = clustering[index*partlen:(index+1)*partlen]

        # select the clustering galaxies in the complete catalogue and the repeat catalogue
        repz,zerr = [],[]
        for j,target in enumerate(TARGETID):
            # select the clustering galaxies with targetid from complete
            if gal == 'LRG':
                poszerr = np.where(complete['RA_REDROCK']==target)[0]
            else:
                poszerr = np.where(targetid_tot==target)[0]

            # according to THING_ID from complete==clustering, select the repetitive measurements
            #print(j,len(complete['THING_ID'][poszerr]))
            #if len(complete['THING_ID'][poszerr])!=1:
            #    print(complete['THING_ID'][poszerr],complete['RA_REDROCK'][poszerr],complete['DEC_REDROCK'][poszerr])
                
            if len(complete['THING_ID'][poszerr])==0:
                zerr.append(np.nan)
                repz.append(np.nan)  
            else:
                pos = np.where(reobs['thids']==np.unique(complete['THING_ID'][poszerr]))[0]
                if len(pos)==1:
                    # if the clustering galaxy has two measurements: zerr = zerr from repeat
                    zerr.append(reobs['zerr'][pos][0])
                    repz.append(reobs['delta_v'][pos][0])
                else:
                    # otherwise, they should be the first 
                    zerr.append((complete['ZERR_REDROCK'][poszerr]*c_kms/(1+complete['Z_REDROCK'][poszerr]))[0])
                    repz.append(np.nan)
                

        return np.array([TARGETID,repz,zerr]).T


    def clusteringSEL(fileN,fileS,zmin,zmax,maxdv):
        if os.path.exists(home+'clustering_zerr/{}_targetid_deltav_zerr_z{}z{}.fits.gz'.format(gal,zmin,zmax)):
            hdu = fits.open(home+'clustering_zerr/{}_targetid_deltav_zerr_z{}z{}.fits.gz'.format(gal,zmin,zmax))
            data = hdu[1].data
            hdu.close()

            LRGtot = np.zeros((len(data['zerr']),3))
            #import pdb;pdb.set_trace()
            for k,name in enumerate(data.columns.names):
                LRGtot[:,k] = data[name]
        else:
            # read clustering data 
            hdu = fits.open(fileN)
            dataN = hdu[1].data
            dataN = dataN[(dataN['Z']>zmin)&(dataN['Z']<zmax)]
            hdu.close()

            hdu = fits.open(fileS)
            dataS = hdu[1].data
            dataS = dataS[(dataS['Z']>zmin)&(dataS['Z']<zmax)]
            hdu.close()
            print('clustering data reading finished.')

            # find clustering galaxies in the repetitive catalogue and the complete catalogue
            LRG = []
            ind = [i for i in range(nthreads)]
            if gal == 'LRG':
                ID = 'ra'
                inputN = dataN['RA']
                inputS = dataS['RA']
            else:
                ID = 'targetid'
                inputN = dataN['PLATE']*1e9+dataN['MJD']*1e4+dataN['FIBERID']
                inputS = dataS['PLATE']*1e9+dataS['MJD']*1e4+dataS['FIBERID']
            datan = [inputN]*nthreads
            datas = [inputS]*nthreads
            
            #import pdb;pdb.set_trace()
            with Pool(processes = nthreads) as p:
                LRG.append(p.starmap(matching,list(zip(ind,datan))))
            print('NGC matching finished.')

            with Pool(processes = nthreads) as p:
                LRG.append(p.starmap(matching,list(zip(ind,datas))))
            print('SGC matching finished.')
            
            for cap in range(2):
                for i in range(nthreads):
                    if (cap == 0)&(i<2):
                        LRGtot = np.vstack((LRG[cap][0],LRG[cap][1]))
                    else:
                        LRGtot = np.vstack((LRGtot,LRG[cap][i]))
            #import pdb;pdb.set_trace()

            cols = []
            formats=['K','D','D']
            for k,colname in enumerate([ID,'delta_v','zerr']):
                cols.append(fits.Column(name=colname,format=formats[k],array=LRGtot[:,k]))
            hdulist = fits.BinTableHDU.from_columns(cols)
            hdulist.writeto(home+'clustering_zerr/{}_targetid_deltav_zerr_z{}z{}.fits.gz'.format(gal,zmin,zmax),overwrite=True)

        bins = np.arange(0,maxdv+1,5)
        densdv,BINS = np.histogram(np.abs(LRGtot[~np.isnan(LRGtot[:,1]),1]),bins=bins)
        denszerr,BINS = np.histogram(np.abs(LRGtot[~np.isnan(LRGtot[:,2]),2]),bins=bins)
        plt.plot((BINS[1:]+BINS[:-1])/2,densdv/sum(densdv),'r',label='$\Delta$ v')
        plt.plot((BINS[1:]+BINS[:-1])/2,denszerr/sum(denszerr),'b',label='ZERR')
        plt.yscale('log')
        plt.legend(loc=1)
        plt.title('{} clustering galaxy |$\Delta$ v| (if exists) v.s. ZERR'.format(gal))
        plt.savefig(home+'clustering_zerr/{}_clustering-deltav_zerr_z{}z{}.png'.format(gal,zmin,zmax))
        plt.close()

    # 
    clusteringSEL(clusteringN,clusteringS,zmin,zmax,maxdv)
    print('z{}z{} selection finished'.format(zmin,zmax))



"""
# test: 
## parallelised
LRG=[]
datas = [dataS['RA'][:2000]]*nthreads
with Pool(processes = nthreads) as p:
    LRG = p.starmap(matching,list(zip(ind,datas))) 
test=[]
for i in range(nthreads):
    if len(LRG[i])!=0:
        for k in range(len(LRG[i])):
            test.append(LRG[i][k])
# loop    
LRGloop=[]
for j,ra in enumerate(dataS['RA'][:2000]):
    pos = np.where(reobs['RA_REDROCK']==ra)[0]
    if len(pos)==2:
        LRGloop.append(pos)

        # whether they equal: yes
np.allclose(np.array(test),np.array(LRGloop))

"""