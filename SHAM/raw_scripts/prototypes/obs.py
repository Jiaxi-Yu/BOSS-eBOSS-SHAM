import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
  
def obs(home,gal,GC,obsname,obsout,rmin,rmax,nbins,zmin,zmax,Om,exist):
    # observation 2pcf calculation in interactive node
    # obs has converted to .dat files
    if exist==False:
        obspc = ascii.read(obsname,format = 'no_header')  # obs pair counts
        mu = (np.linspace(0,1,201)[1:]+np.linspace(0,1,201)[:-1])/2
        
        mon = ((obspc['col3']-2*obspc['col4']+obspc['col5'])/obspc['col5']).reshape(250,200)
        qua = mon * 2.5 * (3 * mu**2 - 1)
        hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        ## use trapz to integrate over mu
        obs0 = np.trapz(mon, dx=1./200, axis=1)
        obs1 = np.trapz(qua, dx=1./200, axis=1)
        obs2 = np.trapz(hexad, dx=1./200, axis=1)
        Table([np.linspace(0,250,251)[:-1],np.linspace(0,250,251)[1:],obs0,obs1,obs2]).write(obsout,format = 'ascii.no_header',delimiter='\t',overwrite=True)
    if exist==True:
        print('The observation 2pcf file already exists.')


