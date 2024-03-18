import numpy as np
import fitsio
from astropy.table import Table, vstack, join
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
gal = sys.argv[1] #LOWZ, CMASS (eBOSS not supported yet)
if gal == 'LRG':
    proj='eBOSS'
    zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
    maxdvs = [235,275,275,300,255,360]
    #zmin = 0.6; zmax = 1.0
    clustering = scratch+'eBOSS_LRG_clustering_data-{}-vDR16.fits'
    z_field = 'Z_REDROCK'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
    caps = ['NGC','SGC']
elif gal == 'CMASS':
    proj='BOSS'
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
    maxdvs = [205,200,235,270]
    #zmin = 0.43; zmax = 0.7
    clustering = scratch+'galaxy_DR12v5_CMASS_{}.fits.gz'
    z_field = 'Z_NOQSO'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
    caps = ['North','South']
elif gal == 'LOWZ':
    proj='BOSS'
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
    maxdvs = [105,140,140]
    #zmin = 0.2; zmax = 0.43
    clustering = scratch+'galaxy_DR12v5_LOWZ_{}.fits.gz'
    z_field = 'Z_NOQSO'
    zerr_field = 'ZERR_REDROCK'
    ra_field = 'RA_REDROCK'
    dec_field = 'DEC_REDROCK'
    caps = ['North','South']
else:
    print("Wrong input")   
import pdb;pdb.set_trace()

output = home+'clustering_zerr/{}_targetid_deltav_zerr.fits.gz'.format(gal)
if not os.path.exists(output):
    # repeat catalogues
    repeatfile = home+'{}-{}_deltav-NGC+SGC.fits.gz'.format(proj,gal)
    reobs = Table(fitsio.read(repeatfile))
    reobs = reobs[(reobs['delta_chi2']>min_dchi2)&(abs(reobs['delta_v'])<1000)]

    # merge clustering in both caps
    data_cap=[]
    for cap in caps:            
        data_cap.append(Table(fitsio.read(clustering.format(cap))))
    clustering = vstack(data_cap)
    clustering['TARGETID'] = clustering['PLATE']*1e9+clustering['MJD']*1e4+clustering['FIBERID']

    # match repeated and clustering   
    tmp = join(clustering,reobs,join_type='left',keys=['TARGETID'])
    assert len(tmp)==len(clustering), "tmp has extra entries after join"
    # remove masked data
    import numpy.ma as ma
    sel = tmp['mag'].mask == False
    # only keep relavent columns
    match = tmp[sel]
    os.makedirs(home+'clustering_zerr',exist_ok = True )
    (match['RA','DEC','z','zerr','delta_v','TARGETID','FIBER2FLUX']).write(output)
    print('file saved:',output)
else:
    print('file exists: ',output)
