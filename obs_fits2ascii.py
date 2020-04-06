from astropy.io import ascii
from astropy.table import Table
import numpy as np
from astropy.io import fits

path = '/media/jiaxi/disk/Master/OneDrive/master_thesis/obs/'
for name in ['NGC_v7_2.dat','SGC_v7_2.dat','NGC_v7_2.ran','SGC_v7_2.ran']:
    hdu = fits.open(path+'eBOSS_LRG_clustering_'+name+'.fits')
    data=hdu[1].data
    ra = data['RA']
    dec= data['DEC']
    z  = data['Z']
    weight_nocoll = data['WEIGHT_SYSTOT']*data['WEIGHT_NOZ']*data['WEIGHT_FKP']
    weight_tot    = data['WEIGHT_SYSTOT']*data['WEIGHT_NOZ']*data['WEIGHT_FKP']*data['WEIGHT_CP']
    
    output = Table([ra,dec,z,weight_nocoll,weight_tot], names=('RA','DEC','Z','WEIGHT_NC','WEIGHT_TOT'))
    if name[-3:]=='dat':
        ascii.write(output,path+name, delimiter='\t')
    else:
        ascii.write(output,path+name+'.dat', delimiter='\t')
    
