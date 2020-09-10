import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
from NGC_SGC import read_xi

Om = 0.31
home      = '/global/cscratch1/sd/jiaxi/master/'
function  = 'xi'
rscale = 'linear' #&'log'
xi0 = [None] * 3
xi2 = [None] * 3
xi4 = [None] * 3
pair_counts = [None]*3
num = [None] * 2
for gal in ['LRG']:
    for i,GC in enumerate(['NGC','SGC','NGC+SGC']):
        if gal == 'ELG':
            obsname  = home+'catalog/nersc_mps_ELG_v7/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
            if rscale == 'linear':
                obs2pcf  = home+'catalog/nersc_mps_ELG_v7/2PCF_'+function+'_'+rscale+'_'+gal+'_'+GC+'.dat'    
            else:
                obs2pcf  = home+'catalog/nersc_zbins_wp_mps_ELG/2PCF_'+function+'_'+rscale+'_'+gal+'_'+GC+'.dat' 
        else:
            obsname  = home+'catalog/nersc_mps_LRG_v7_2/pair_counts_s-mu_pip_eBOSS+SEQUELS_'+gal+'_'+GC+'_v7_2.dat'
            if rscale == 'linear':
                obs2pcf  = home+'catalog/nersc_mps_LRG_v7_2/2PCF_'+function+'_'+rscale+'_'+gal+'_'+GC+'.dat'    
            else:
                obs2pcf  = home+'catalog/nersc_zbins_wp_mps_LRG/2PCF_'+function+'_'+rscale+'_'+gal+'_'+GC+'.dat' 
 
        # observations in NGC,SGC and NGC+SGC
        obspc = ascii.read(obsname,format = 'no_header')
        mu = (np.linspace(0,1,201)[1:]+np.linspace(0,1,201)[:-1])/2
        mon = ((obspc['col3']-2*obspc['col4']+obspc['col5'])/obspc['col5']).reshape(250,200)
        qua = mon * 2.5 * (3 * mu**2 - 1)
        hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        ## use trapz to integrate over mu
        #** modification made
        obs0 = np.sum(mon,axis=1)/200.
        obs1 = np.sum(qua,axis=1)/200.
        obs2 = np.sum(hexad,axis=1)/200.
        Table([np.linspace(0,250,251)[:-1],np.linspace(0,250,251)[1:],obs0,obs1,obs2]).write(obs2pcf,format = 'ascii.no_header',delimiter='\t',overwrite=True)


