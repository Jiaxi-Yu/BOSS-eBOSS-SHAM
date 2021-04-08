from astropy.io import fits
import numpy as np
import sys

zw = {}

if sys.argv[1] == 'eBOSS':
    for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
        f = open(gal+'zeff.txt','w')
        f.write('# zmin zmax zeff Ngal \n')
        for GC in ['NGC','SGC']:
            filename = '/global/cscratch1/sd/jiaxi/master/catalog/eBOSS_{}_clustering_{}_{}.dat.fits'.format(gal,GC,ver)
            hdu = fits.open(filename)
            zw[gal+'_'+GC+'_z']=hdu[1].data['Z']
            zw[gal+'_'+GC+'_w']=hdu[1].data['WEIGHT_FKP']*hdu[1].data['WEIGHT_SYSTOT']*hdu[1].data['WEIGHT_CP']*hdu[1].data['WEIGHT_NOZ']
        for zmin,zmax in zip([0.6,0.7,0.8],[0.8,0.9,1.0]):
            sel1 = (zw[gal+'_NGC_z']>zmin)&(zw[gal+'_NGC_z']<=zmax)
            sel2 = (zw[gal+'_SGC_z']>zmin)&(zw[gal+'_SGC_z']<=zmax)

            zeff = (sum(zw[gal+'_NGC_z'][sel1]*zw[gal+'_NGC_w'][sel1]**2)+sum(zw[gal+'_SGC_z'][sel2]*zw[gal+'_SGC_w'][sel2]**2))/(sum(zw[gal+'_NGC_w'][sel1]**2)+sum(zw[gal+'_SGC_w'][sel2]**2))
            f.write('{} {} {} {} \n'.format(zmin,zmax,zeff,len(zw[gal+'_NGC_z'][sel1])+len(zw[gal+'_SGC_z'][sel2])))

        f.close()
else:
    ver = 'DR12v5'
    root = '/home/jiaxi/Desktop/BOSS_clustering/'
    for gal in ['CMASS','LOWZ']:
        f = open(root+gal+'zbins.txt','w')
        f.write('# zmin zmax zeff Ngal ngal(e-4)\n')
        for GC in ['North','South']:
            filename = root+'galaxy_{}_{}_{}.fits.gz'.format(ver,gal,GC)
            hdu = fits.open(filename)
            zw[gal+'_'+GC+'_z']=hdu[1].data['Z']
            if gal == 'LOWZ':
                zmins = [0.15,0.2, 0.33,0.2]
                zmaxs = [0.2, 0.33,0.43,0.43]
                zw[gal+'_'+GC+'_w']=hdu[1].data['WEIGHT_FKP']
            elif gal == 'CMASS':
                zmins = [0.43,0.51,0.57,0.43]
                zmaxs = [0.51,0.57,0.70,0.70]
                zw[gal+'_'+GC+'_w']=hdu[1].data['WEIGHT_FKP']*hdu[1].data['WEIGHT_SYSTOT']*hdu[1].data['WEIGHT_CP']*hdu[1].data['WEIGHT_NOZ']
                
        for zmin,zmax in zip(zmins,zmaxs):
            sel1 = (zw[gal+'_North_z']>zmin)&(zw[gal+'_North_z']<=zmax)
            sel2 = (zw[gal+'_South_z']>zmin)&(zw[gal+'_South_z']<=zmax)

            zeff = (sum(zw[gal+'_North_z'][sel1]*zw[gal+'_North_w'][sel1]**2)+sum(zw[gal+'_South_z'][sel2]*zw[gal+'_South_w'][sel2]**2))/(sum(zw[gal+'_North_w'][sel1]**2)+sum(zw[gal+'_South_w'][sel2]**2))
            f.write('{} {} {} {} \n'.format(zmin,zmax,zeff,len(zw[gal+'_North_z'][sel1])+len(zw[gal+'_South_z'][sel2])))

    f.close()
