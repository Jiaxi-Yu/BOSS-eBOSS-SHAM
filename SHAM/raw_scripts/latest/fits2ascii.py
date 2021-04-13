#!/usr/bin/env python3
from astropy.io import ascii
from astropy.table import Table
import numpy as np
from astropy.io import fits
import sys

if sys.argv[1] == 'LOWZ':
    titles = ['galaxy','random0']
    names = ['LOWZ_North','LOWZ_South']
    path = '/home/astro/jiayu/Desktop/SHAM/catalog/BOSS_clustering/'
elif sys.argv[1] == 'CMASS': 
    titles = ['galaxy','random0']
    names = ['CMASS_North','CMASS_South']
    path = '/home/astro/jiayu/Desktop/SHAM/catalog/BOSS_clustering/'
elif sys.argv[1] == 'CMASSLOWZ': 
    titles = ['random_20x']#['galaxy','random_20x']
    names = ['CMASSLOWZTOT_North','CMASSLOWZTOT_South']
    path = '/global/cscratch1/sd/jiaxi/SHAM/catalog/BOSS_data/'
elif sys.argv[1] == 'eBOSS':
    titles = ['eBOSS_LRG_clustering_']
    names = ['NGC_v7_2.dat','SGC_v7_2.dat','NGC_v7_2.ran','SGC_v7_2.ran']
    path = '/media/jiaxi/disk/Master/OneDrive/master_thesis/obs/'

for title in titles:
    for i,name in enumerate(names):
        if sys.argv[1] == 'eBOSS':
            hdu = fits.open(path+title+name+'.fits')
        else:
            hdu = fits.open(path+title+'_DR12v5_'+name+'.fits.gz')
        data=hdu[1].data
        ra = data['RA']
        dec= data['DEC']
        z  = data['Z']
        if sys.argv[1] == 'eBOSS':
            weight_tot    = data['WEIGHT_SYSTOT']*data['WEIGHT_NOZ']*data['WEIGHT_CP']*data['WEIGHT_FKP']
        elif ((sys.argv[1] == 'CMASS')&(title=='galaxy'))|((sys.argv[1] == 'CMASSLOWZTOT')&(title=='galaxy')): 
            weight_tot    = data['WEIGHT_SYSTOT']*(data['WEIGHT_NOZ']+data['WEIGHT_CP']-1)*data['WEIGHT_FKP']
        else:   
            weight_tot    = data['WEIGHT_FKP']

        output = Table([ra,dec,z,weight_tot], names=('RA','DEC','Z','WEIGHT_TOT'))
        print(name+' finish')
        if sys.argv[1] == 'eBOSS': 
            if name[-3:]=='dat':
                ascii.write(output,path+name, delimiter='\t')
            else:
                ascii.write(output,path+name+'.dat', delimiter='\t')
        else:
            ascii.write(output,path+title+'_DR12v5_'+name+'.dat', delimiter='\t')
        
