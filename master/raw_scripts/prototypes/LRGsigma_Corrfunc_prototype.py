import time
time_start=time.time()
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from Corrfunc.theory.DD import DDsmu
from iminuit import Minuit

import os


# variables
dir = '/global/cscratch1/sd/jiaxi/master/'
rscale = 'linear' # 'log'
mockfile = 
halofile = 
obsfile  = 
rmin=0
rmax=50
nbins=50
boxsize = 2500
nthread = 64
LRGnum  = 5468750
autocorr=1

# bins array
if rscale='linear':
	bins=np.linspace(rmin,rmax,nbins+1)
if rscale='log':
	bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
# RR array from the bins arrany
RR_counts = np.zeros(nbins)
for b in range(nbins):
	RR_counts[b] = 4*np.pi/3*(bins[b+1]**3-bins[b]**3)/2/(boxsize**3)

# covmatrix(mockdir, rmin,rmax,zmin,zmax,Om):(redshift space)
hdu = fits.open(mockfile)
mock_tmp= hdu[1].data
covmono = mocks_tmp['covmono']
covquadru = mocks_tmp['covquadru']
covhexa = mocks_tmp['covhexa']
hdu.close()
obs = fits.open(obsfile)

# read the halo catalogue
halo = fits.open(halofile)
data = halo[1].data
vpeak = data['vpeak']
halo.close()
# fit the best sigma for the LRG catalogue and time the process
np.random.seed(0)
datac = np.copy(data)

## find the best parameters
def chi2(Sigma):
	### create the LRG catalogues
	datac['vpeak'] *= 1+np.random.normal(scale=Sigma,size=len(datac['vpeak']))
	sort_scat = datac[datac['vpeak'].argsort()]
	LRGscat = sort_scat[::-1][:LRGnum]
	DD_counts = DDsmu(autocorr, nthreads, bins,LRGscat['X'],LRGscat['Y'],LRGscat['Z'],periodic=True, verbose=True,boxsize=boxsize)
	DD = DD_counts['npairs']/(len(X))**2
	
	### calculate the 2pcf and the multipoles
	mono = DD/RR-1
	quadru = mono*

	### calculate the covariance, residuals and chi2
	res = obs-model
	resTcov = res.dot(cov)
	return resTcov.dot(res)

# chi2 minimise
sigma = Minuit(chi2,Sigma=0.3)
sigma.migrad()  # run optimiser
print(sigma.values) 
time_end=time.time()
print('Creating LRG catalogue costs',time_end-time_start,'s')
print('the best LRG distribution sigma is ',sigma.values)


