import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
  
def obs(home,gal,GC,obsname,randname,obsout,rmin,rmax,nbins,zmin,zmax,Om,exist):
    # observation 2pcf calculation in interactive node
    # obs has converted to .dat files
	if (gal=='LRG')&(exist==False):
		print('observation 2pcf calculation:') 
		print('./FCFC/2pcf -c ./FCFC/fcfc.conf -j 1 -a '+str(rmin)+' -b '+str(rmax)+' -n '+str(nbins)+' --data-z-min '+str(zmin)+' --data-z-max '+str(zmax)+' --rand-z-min '+str(zmin)+' --rand-z-max '+str(zmax)+' -m '+str(Om)+' --data '+obsname[:-5]+' --rand '+randname[:-5]+' --dd '+home+'2PCF/obs/LRG_'+GC+'.dd --dr '+home+'2PCF/obs/LRG_'+GC+'.dr --rr '+home+'2PCF/obs/LRG_'+GC+'.rr --output '+obsout)
		os.system('./FCFC/2pcf -c ./FCFC/fcfc.conf -j 1 -a '+str(rmin)+' -b '+str(rmax)+' -n '+str(nbins)+' --data-z-min '+str(zmin)+' --data-z-max '+str(zmax)+' --rand-z-min '+str(zmin)+' --rand-z-max '+str(zmax)+' -m '+str(Om)+' --data '+obsname[:-5]+' --rand '+randname[:-5]+' --dd '+home+'2PCF/obs/LRG_'+GC+'.dd --dr '+home+'2PCF/obs/LRG_'+GC+'.dr --rr '+home+'2PCF/obs/LRG_'+GC+'.rr --output '+obsout)
	elif (gal=='ELG')&(exist==False):
                obspc = ascii.read(obsname,format = 'no_header')  # obs pair counts
                mu = np.linspace(0,1,200)
                mon = ((obspc['col3']-2*obspc['col4']+obspc['col5'])/obspc['col5']/200).reshape(250,200)
                qua = mon * 2.5 * (3 * mu**2 - 1)
                hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
		## use trapz to integrate over mu
                obs0 = np.trapz(mon, dx=1./200, axis=1)
                obs1 = np.trapz(qua, dx=1./200, axis=1)
                obs2 = np.trapz(hexad, dx=1./200, axis=1)
                Table([np.linspace(0,250,251)[:-1],np.linspace(0,250,251)[1:],obs0,obs1,obs2]).write(obsout,format = 'ascii',delimiter='\t',overwrite=True)
	else:
                print('The observation 2pcf file already exists.')


