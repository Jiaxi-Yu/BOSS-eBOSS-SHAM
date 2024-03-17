import numpy as np
import fitsio
from astropy.table import Table, vstack
from multiprocessing import Pool 
from itertools import repeat
import pylab as plt
import sys
import os

c_kms = 299792.
min_dchi2 = 9
fc_limit = (62/3600)**2 # fibre collision limit

# load the total catalogue and the repetitive samples
scratch = os.environ['SCRATCH']+'/SHAM/catalog/SDSS_data/'
home = os.environ['HOME']+'/SDSS_redshift_uncertainty/Vsmear-reproduce/'
gal = sys.argv[1]
GC = sys.argv[2]
nthreads = int(sys.argv[3])
if gal == 'LRG':
    proj='eBOSS'
    zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
    maxdvs = [235,275,275,300,255,360]
    #zmin = 0.6; zmax = 1.0
    clusteringN = scratch+'eBOSS_LRG_clustering_NGC_v7_2.dat.fits'
    clusteringS = scratch+'eBOSS_LRG_clustering_SGC_v7_2.dat.fits'
    z_field = 'Z_REDROCK'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
elif gal == 'CMASS':
    proj='BOSS'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
    maxdvs = [205,200,235,270]
    #zmin = 0.43; zmax = 0.7
    clusteringN = scratch+'galaxy_DR12v5_CMASS_North.fits.gz'
    clusteringS = scratch+'galaxy_DR12v5_CMASS_South.fits.gz'
    z_field = 'Z_NOQSO'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
elif gal == 'LOWZ':
    proj='BOSS'
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
    maxdvs = [105,140,140]
    #zmin = 0.2; zmax = 0.43
    clusteringN = scratch+'galaxy_DR12v5_LOWZ_North.fits.gz'
    clusteringS = scratch+'galaxy_DR12v5_LOWZ_South.fits.gz'
    z_field = 'Z_NOQSO'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
else:
    print("Wrong input")   

completefile = scratch+'spAll-zbest-v5_13_0.fits'
complete = Table(fitsio.read(completefile))
complete['TARGETID'] = complete['PLATE']*1e9+complete['MJD']*1e4+complete['FIBERID']
print('the complete sample reading finished.')

repeatfile = home+'{}-{}_deltav-NGC+SGC.fits.gz'.format(proj,gal)
reobs_raw = Table(fitsio.read(repeatfile))
    
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    reobs = reobs_raw[(reobs_raw['delta_chi2']>min_dchi2)&(abs(reobs_raw['delta_v'])<1000)&(zmin<reobs_raw['z'])&(reobs_raw['z']<zmax)]
    print('the repetitive sample sample reading finished.')

    # the matching function
    def matching(index,clustering):
        #slice the clustering catalogue in nthread parts
        partlen = len(clustering)//nthreads
        if index == nthreads-1:
            TARGETID = clustering[index*partlen:,:]
        else:
            TARGETID = clustering[index*partlen:(index+1)*partlen,:]

        # select the clustering galaxies in the complete catalogue and the repeat catalogue
        repz,zerr,imag,z = [],[],[],[]
        for j,target in enumerate(TARGETID):
            # select the clustering galaxies with targetid from complete
            if gal == 'LRG':
                poszerr = np.where(complete[ra_field]==target[0])[0]
            else:
                poszerr = np.where(complete['TARGETID']==target[0])[0]

            # according to THING_ID from complete==clustering, select the repetitive measurements
            if len(complete['THING_ID'][poszerr])==0:
                if gal == 'LRG':
                    zdiff = (complete[z_field]-target[2])/(1+target[2])*c_kms
                    distance = (complete[ra_field]-target[0])**2+(complete[dec_field]-target[1])**2
                else:
                    zdiff = (complete[z_field]-target[3])/(1+target[3])*c_kms
                    distance = (complete[ra_field]-target[1])**2+(complete[dec_field]-target[2])**2

                poszerr0 = np.where((distance<fc_limit)&(zdiff<1000))[0] 
                if len(poszerr0)==0:
                    print('warning! clustering no.{} ({}) has no matching TARGETID nor position'.format(index*partlen+j,target[0]))
                    zerr.append(np.nan)
                    repz.append(np.nan)  
                    imag.append(np.nan)
                    z.append(np.nan)
                else:
                    pos = np.where(reobs['thids']==np.unique(complete['THING_ID'][poszerr0]))[0]
                    if len(pos) ==1:
                        # if the clustering galaxy has two measurements: zerr = zerr from repeat
                        zerr.append(reobs['zerr'][pos][0])
                        repz.append(reobs['delta_v'][pos][0])
                        imag.append(reobs['mag'][pos][0])
                        z.append(reobs['z'][pos][0])                        
                    else:
                        # otherwise, they should be the first 
                        zerr.append((complete[zerr_field][poszerr0]*c_kms/(1+complete[z_field][poszerr0]))[0])
                        repz.append(np.nan)                    
                        imag.append(22.5 - 2.5 * np.log10((complete['SPECTROFLUX'][poszerr0][0])[3]))
                        z.append(complete[z_field][poszerr0][0])                    
            else:
                pos = np.where(reobs['thids']==np.unique(complete['THING_ID'][poszerr]))[0]
                if len(pos)==1:
                    # if the clustering galaxy has two measurements: zerr = zerr from repeat
                    zerr.append(reobs['zerr'][pos][0])
                    repz.append(reobs['delta_v'][pos][0])
                    imag.append(reobs['mag'][pos][0])
                    z.append(reobs['z'][pos][0])                        
                else:
                    # otherwise, they should be the first 
                    zerr.append((complete[zerr_field][poszerr]*c_kms/(1+complete[z_field][poszerr]))[0])
                    repz.append(np.nan)
                    imag.append(22.5 - 2.5 * np.log10((complete['SPECTROFLUX'][poszerr][0])[3]))
                    z.append(complete[z_field][poszerr][0])    
        return np.array([TARGETID[:,0],z,repz,zerr,imag]).T

    def clusteringSEL(filename,cap,zmin,zmax,maxdv):
        if os.path.exists(home+'clustering_zerr/{}_targetid_deltav_zerr-{}.fits.gz'.format(gal,cap)):
            data = Table(fitsio.read(home+'clustering_zerr/{}_targetid_deltav_zerr-{}.fits.gz'.format(gal,cap)))

            LRGtot = np.zeros((len(data['zerr']),3))
            #import pdb;pdb.set_trace()
            for k,name in enumerate(data.columns.names):
                LRGtot[:,k] = data[name]
        else:
            # read clustering data 
            dataN = Table(fitsio.read(filename))
            print('clustering data reading finished.')

            # find clustering galaxies in the repetitive catalogue and the complete catalogue
            ind = [i for i in range(nthreads)]
            inputN = Table()
            for key in ['RA','DEC','Z']:
                inputN[key] = dataN[key]*1
            if gal == 'LRG':
                ID = 'ra'
            else:
                ID = 'targetid'
                inputN['TARGETID'] = [dataN['PLATE']*1e9+dataN['MJD']*1e4+dataN['FIBERID']]

            # NGC matching
            LRG = []; LRGtot = []
            datan = [inputN]*nthreads
            with Pool(processes = nthreads) as p:
                LRG = p.starmap(matching,list(zip(ind,datan)))
            print('{}GC matching finished.'.format(cap))  
            LRGtot = vstack(LRG)
            """
            ## merge threads
            for i in range(nthreads):
                if (i<2):
                    LRGtot = np.vstack((LRG[0],LRG[1]))
                else:
                    LRGtot = np.vstack((LRGtot,LRG[i]))
            """
            ## save NGC
            cols = []
            formats=['K','D','D','D','D']
            for k,colname in enumerate([ID,'delta_v','z','zerr','imag']):
                cols.append(fits.Column(name=colname,format=formats[k],array=LRGtot[:,k]))
            hdulist = fits.BinTableHDU.from_columns(cols)
            hdulist.writeto(home+'clustering_zerr/{}_targetid_deltav_zerr-{}.fits.gz'.format(gal,cap),overwrite=True)

    # 
    if GC =='N':
        clusteringSEL(clusteringN,GC,maxdv)
    elif GC =='S':
        clusteringSEL(clusteringS,GC,maxdv)
    print('{} selection finished'.format(GC))



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
