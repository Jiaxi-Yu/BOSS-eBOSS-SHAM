import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
  
def obsz(obsname,obsout,exist):
    if exist==False:
        obsdd = ascii.read(obsname+'.dd',format = 'no_header')['col6']  # obs pair counts
        obsdr = ascii.read(obsname+'.dr',format = 'no_header')['col6']  # obs pair counts
        obsrr = ascii.read(obsname+'.rr',format = 'no_header')['col6']  # obs pair counts
        mu = (np.linspace(0,1,121)[1:]+np.linspace(0,1,121)[:-1])/2
        mon = ((obsdd-2*obsdr+obsrr)/obsrr).reshape(200,120)
        qua = mon * 2.5 * (3 * mu**2 - 1)
        hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        ## use trapz to integrate over mu
        obs0 = np.trapz(mon, dx=1./120, axis=1)
        obs1 = np.trapz(qua, dx=1./120, axis=1)
        obs2 = np.trapz(hexad, dx=1./120, axis=1)
        Table([np.linspace(0,200,201)[:-1],np.linspace(0,200,201)[1:],obs0,obs1,obs2]).write(obsout,format = 'ascii.no_header',delimiter='\t',overwrite=True)
    if exist==True:
        print('The observation 2pcf file already exists.')  


