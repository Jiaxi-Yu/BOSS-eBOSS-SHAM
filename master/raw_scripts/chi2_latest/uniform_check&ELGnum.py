import matplotlib 
matplotlib.use('agg')
import time
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob

# validate the uniform parameter used in sigma_mpi
home      = '/global/cscratch1/sd/jiaxi/master/'
filename = home+'catalog/randnum/random_10.fits.gz'
uniform =  fits.open(filename)[1].data
sigma = 0.3
rand = np.append(sigma*np.sqrt(-2*np.log(uniform['col0']))*np.cos(2*np.pi*uniform['col1']),sigma*np.sqrt(-2*np.log(uniform['col0']))*np.sin(2*np.pi*uniform['col1'])) 
nums,bin = np.histogram(rand,bins=200,range=(-1.5,1.5))
s = (bin[:-1]+bin[1:])/2
def gaussian(x,*par):
	return par[0]*np.exp(-0.5*(x-par[1])**2/par[2]**2)

a,b = curve_fit(gaussian,s,nums,p0=(1200000,0,0.2))

fig,ax = plt.subplots()
ax.hist(rand,bins=200,range=(-1.5,1.5),color='r',label='rand distr.')
ax.plot(s,gaussian(s,*a),'k-',label='Gaussian fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(a))
plt.legend(loc=0)
plt.savefig('uniform-trans_validate.png')


# calculate the number of ELG selected in the UNIT catalogue
area = np.array([177.5,240,237.5,131.9])
areadeg = np.array([417.5,369.4])

z=np.linspace(0.6,1.1,21)
c=299792.458
Om=0.31
dfin = c/100/np.sqrt(Om*(1+0.6)**3+1-Om)
dint = c/100/np.sqrt(Om*(1+1.1)**3+1-Om)
V = 4/3*np.pi*(dfin**3-dint**3)*np.sum(areadeg)/36000

for i,GC in enumerate(['NGC','SGC']):
	hdu = fits.open('/media/jiaxi/disk/Master/obs/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits')[1].data
	galnum=np.sum(hdu['WEIGHT_SYSTOT']*hdu['WEIGHT_CP']*hdu['WEIGHT_NOZ']*hdu['WEIGHT_FKP'])
	print(GC+' have {:2.7} galaxies'.format(galnum))
	print('its number density is {:2.7} gal/(Mpc)**3'.format(galnum/V))

## results:
'''
NGC have 45367.24 galaxies
its number density is 8.526405e-05 gal/(Mpc)**3
SGC have 44901.41 galaxies
its number density is 8.438856e-05 gal/(Mpc)**3
difference is within 1%

UNIT ELG numbers
NGC:85264.05
SGC:84388.56

'''







