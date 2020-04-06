import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob
  
def obs(home,GC,obsname,randname,rmin,rmax,nbins,zmin,zmax,Om,exist):
    # observation 2pcf calculation in interactive node
    # obs has converted to .dat files
	if exist==False:
		print('observation 2pcf calculation:') 
		print('./FCFC/2pcf -c ./FCFC/fcfc.conf -j 1 -a '+str(rmin)+' -b '+str(rmax)+' -n '+str(nbins)+' --data-z-min '+str(zmin)+' --data-z-max '+str(zmax)+' --rand-z-min '+str(zmin)+' --rand-z-max '+str(zmax)+' -m '+str(Om)+' --data '+home+'catalog/'+obsname[:-5]+' --rand '+home+'catalog/'+randname[:-5]+' --dd '+home+'2PCF/obs/LRG_'+GC+'.dd --dr '+home+'2PCF/obs/LRG_'+GC+'.dr --rr '+home+'2PCF/obs/LRG_'+GC+'.rr --output '+home+'2PCF/obs/LRG_'+GC+'.dat')
		os.system('./FCFC/2pcf -c ./FCFC/fcfc.conf -j 1 -a '+str(rmin)+' -b '+str(rmax)+' -n '+str(nbins)+' --data-z-min '+str(zmin)+' --data-z-max '+str(zmax)+' --rand-z-min '+str(zmin)+' --rand-z-max '+str(zmax)+' -m '+str(Om)+' --data '+home+'catalog/'+obsname[:-5]+' --rand '+home+'catalog/'+randname[:-5]+' --dd '+home+'2PCF/obs/LRG_'+GC+'.dd --dr '+home+'2PCF/obs/LRG_'+GC+'.dr --rr '+home+'2PCF/obs/LRG_'+GC+'.rr --output '+home+'2PCF/obs/LRG_'+GC+'.dat')
	else:
		print('The observation 2pcf file already exists.')

#LRG = fits.open(home+'catalog/'+obsname)
#z   = LRG[1].data['Z'][(LRG[1].data['Z']>zmin)&(LRG[1].data['Z']<zmax)]
#LRG.close()

#return z.shape[0]
