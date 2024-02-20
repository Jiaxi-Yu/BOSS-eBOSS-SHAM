from astropy.io import fits
import numpy as np
import sys

zw = {}
root =  '/home/jiaxi/Desktop/data_archive/' #'/global/cscratch1/sd/jiaxi/SHAM/catalog/BOSS_data/' #

if sys.argv[1] == 'eBOSS':
    for gal,ver in zip(['LRG'],['v7_2']):#zip(['LRG','ELG'],['v7_2','v7']):
        f = open(gal+'zeff.txt','w')
        f.write('# zmin zmax zeff Ngal \n')
        for GC in ['NGC','SGC']:
            filename = root+'eBOSS_clustering/eBOSS_{}_clustering_{}_{}.dat.fits'.format(gal,GC,ver)
            hdu = fits.open(filename)
            zw[gal+'_'+GC+'_z']=hdu[1].data['Z']
            zw[gal+'_'+GC+'_w']=hdu[1].data['WEIGHT_FKP']*hdu[1].data['WEIGHT_SYSTOT']*hdu[1].data['WEIGHT_CP']*hdu[1].data['WEIGHT_NOZ']
            if gal == 'LRG':
                zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
                zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
            else:
                zmins = [0.6,0.7,0.8,0.9,0.6]
                zmaxs = [0.8,0.9,1.0,1.1,1.1]
        for zmin,zmax in zip(zmins,zmaxs):
            sel1 = (zw[gal+'_NGC_z']>zmin)&(zw[gal+'_NGC_z']<=zmax)
            sel2 = (zw[gal+'_SGC_z']>zmin)&(zw[gal+'_SGC_z']<=zmax)

            zeff = (sum(zw[gal+'_NGC_z'][sel1]*zw[gal+'_NGC_w'][sel1]**2)+sum(zw[gal+'_SGC_z'][sel2]*zw[gal+'_SGC_w'][sel2]**2))/(sum(zw[gal+'_NGC_w'][sel1]**2)+sum(zw[gal+'_SGC_w'][sel2]**2))
            f.write('{} {} {} {} \n'.format(zmin,zmax,zeff,len(zw[gal+'_NGC_z'][sel1])+len(zw[gal+'_SGC_z'][sel2])))

        f.close()
else:
    ver = 'DR12v5'
    for gal in ['CMASSLOWZTOT']:#,'CMASS','LOWZ']:
        f = open(root+'clustering_BOSS/'+gal+'zbins.txt','a')
        f.write('# zmin zmax zeff Ngal ngal(e-4)\n')
        for GC in ['North','South']:
            filename = root+'clustering_BOSS/'+'galaxy_{}_{}_{}.dat'.format(ver,gal,GC)
            X,Y,Z,w_tot = np.loadtxt(filename,unpack=True,skiprows=1)
            zw[gal+'_'+GC+'_z'] = np.copy(Z)
            zw[gal+'_'+GC+'_w'] = np.copy(w_tot)
            if gal == 'LOWZ':
                zmins = [0.15,0.2, 0.33,0.2]
                zmaxs = [0.2, 0.33,0.43,0.43]
            elif gal == 'CMASS':
                zmins = [0.43,0.51,0.57,0.43]
                zmaxs = [0.51,0.57,0.70,0.70]
            elif gal == 'CMASSLOWZTOT':
                zmins = [0.2, 0.4]
                zmaxs = [0.75,0.6]
                
        for zmin,zmax in zip(zmins,zmaxs):
            sel1 = (zw[gal+'_North_z']>zmin)&(zw[gal+'_North_z']<=zmax)
            sel2 = (zw[gal+'_South_z']>zmin)&(zw[gal+'_South_z']<=zmax)

            zeff = (sum(zw[gal+'_North_z'][sel1]*zw[gal+'_North_w'][sel1]**2)+sum(zw[gal+'_South_z'][sel2]*zw[gal+'_South_w'][sel2]**2))/(sum(zw[gal+'_North_w'][sel1]**2)+sum(zw[gal+'_South_w'][sel2]**2))
            f.write('{} {} {} {} \n'.format(zmin,zmax,zeff,len(zw[gal+'_North_z'][sel1])+len(zw[gal+'_South_z'][sel2])))

    f.close()
