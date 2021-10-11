#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
import os
from kcorrection import kcorr

home = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
# photometric info
spall = fitsio.read(home+'photoPosPlate_dr16clustering.fits.gz')
ta = Table(spall)

for gal in zip(['LOWZ','CMASS']):
    for GC in ['North','South']:
        # clustering reading
        redrock = fitsio.read(home+'BOSS_data/galaxy_DR12v5_{}_{}.fits.gz'.format(gal,GC))
        tc = Table(redrock)
        # macthing photo and clustering
        tca = join(tc,ta, keys=['RUN', 'CAMCOL','ID','FIELD'],join_type='left')
        flux = tca['CMODELFLUX']
        fluxivar = tca['CMODELFLUX_IVAR']
        import pdb;pdb.set_trace()
        tca['flux055'] = np.zeros(len(flux))
        for i in range(len(flux)):
            tca['flux055'][i] = kcorr(tca['Z'][i],tca['CMODELFLUX'][i],tca['CMODELFLUX_IVAR'][i])
        tca.write(home+'/{}_{}_flux.fits.gz'.format(gal,GC), format='fits', overwrite=True)
