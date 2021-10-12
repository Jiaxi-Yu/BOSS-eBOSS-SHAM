#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
import os
from kcorrection import kcorr
import kcorrect 
kcorrect.load_templates()
kcorrect.load_filters()

home = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
# photometric info
spall = fitsio.read(home+'photoPosPlate_dr16clustering.fits.gz')
ta = Table(spall)

for gal in ['LOWZ','CMASS']:
    if not os.path.exists(home+'/{}_{}_flux.fits.gz'.format(gal,'South')):
        for GC in ['North','South']:
            # clustering reading
            redrock = fitsio.read(home+'BOSS_data/galaxy_DR12v5_{}_{}.fits.gz'.format(gal,GC))
            tc = Table(redrock)
            # macthing photo and clustering
            tca = join(tc,ta, keys=['RUN', 'CAMCOL','ID','FIELD'],join_type='left')
            tca['flux055'] = tca['CMODELFLUX']*0
            tca['gi'] = tca['RA']*-99
            for i in range(len(tca)):
                tca['flux055'][i] = kcorr(tca['Z'][i],tca['CMODELFLUX'][i],tca['CMODELFLUX_IVAR'][i])
                tca['gi'][i] = -2.5*np.log10(tca['flux055'][i][1]/tca['flux055'][i][3])
            tca.write(home+'/{}_{}_flux.fits.gz'.format(gal,GC), format='fits', overwrite=True)
    else:
        if gal =='CMASS':
            zmins = [0.43,0.51,0.57,0.43]
            zmaxs = [0.51,0.57,0.7,0.7] 
        elif gal == 'LOWZ':
            zmins = [0.2, 0.33,0.2]
            zmaxs = [0.33,0.43,0.43]            
        hdu = fits.open(home+'{}_{}_flux.fits.gz'.format(gal,'North'))
        dataN = hdu[1].data;hdu.close()
        hdu = fits.open(home+'{}_{}_flux.fits.gz'.format(gal,'South')) 
        dataS = hdu[1].data;hdu.close()
        weightN = dataN['WEIGHT_FKP']*dataN['WEIGHT_SYSTOT']*dataN['WEIGHT_CP']*dataN['WEIGHT_NOZ']
        weightS = dataS['WEIGHT_FKP']*dataS['WEIGHT_SYSTOT']*dataS['WEIGHT_CP']*dataS['WEIGHT_NOZ']
        for zmin,zmax in zip(zmins,zmaxs):
            selN = (dataN['Z']<zmax)&(dataN['Z']>=zmin)
            selS = (dataS['Z']<zmax)&(dataS['Z']>=zmin) 
            lentot = len(dataN[selN])+len(dataS[selS])
            lenred = len(dataN[selN&(dataN['gi']>=2.35)])+len(dataS[selS&(dataS['gi']>=2.35)])
            lenblue= len(dataN[selN&(dataN['gi']<2.35)])+len(dataS[selS&(dataS['gi']<2.35)])
            print('{} galaxies in {} at {}<z<{}, red = {}, blue = {}, red/blue = {}'.format(lentot,gal,zmin,zmax,lenred,lenblue,lenred/lenblue))
            lentotw = sum(weightN[selN])+sum(weightS[selS])
            lenredw = sum(weightN[selN&(dataN['gi']>=2.35)])+sum(weightS[selS&(dataS['gi']>=2.35)])
            lenbluew= sum(weightN[selN&(dataN['gi']<2.35)])+sum(weightS[selS&(dataS['gi']<2.35)])
            print('{} weighted galaxies in {} at {}<z<{}, red = {}, blue = {}, red/blue = {}'.format(lentotw,gal,zmin,zmax,lenredw,lenbluew,lenredw/lenbluew))
