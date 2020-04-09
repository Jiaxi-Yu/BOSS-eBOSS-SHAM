import time
time_start=time.time()
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from iminuit import Minuit
import os


# variables
dir = '/global/cscratch1/sd/jiaxi/master/'
rscale = 'linear' # 'log' 
mockfile = 
halofile = 
rmin=0.1
rmax=200
nbins=40
LRGnum  = 

# read the observation 2pcf results and mocks
hdu = fits.open(mockfile)
mock_tmp= hdu[1].data
covmono = mocks_tmp['covmono']
covquadru = mocks_tmp['covquadru']
covhexa = mocks_tmp['covhexa']

# read the halo catalogue
halo = fits.open(halofile)
data = halo[1].data
vpeak = data['vpeak']

# fit the best sigma for the LRG catalogue and time the process
np.random.seed(0)
datac = np.copy(data)

## find the best parameters
def chi2(Sigma):
	### create the LRG catalogues
	datac['vpeak'] *= 1+np.random.normal(scale=Sigma,size=len(datac['vpeak']))
	sort_scat = datac[datac['vpeak'].argsort()]
	LRGscat = sort_scat[::-1][:LRGnum]
	scat = Table([LRGscat['X'],LRGscat['Y'],LRGscat['Z']], names=('x','y','z'))
	ascii.write(scat,dir+'chi2/LRGcat/sigma'+str(Sigma)+'.dat', delimiter='\t')
	model = Table.read(dir+'chi2/LRG2pcf/sigma'+str(Sigma)+'.dat',format="ascii.no_header")
	
	### calculate the 2pcf and read it
	print('2pcf -c fcfc.conf -j 0 -q bins.dat --data '+dir+'chi2/LRGcat/sigma'+str(Sigma)+'.dat --dd '+dir+'chi2/LRG2pcf/sigma'+str(Sigma)+'.dd --output '+dir+'chi2/LRG2pcf/sigma'+str(Sigma)+'.dat --force 1')
	os.system('2pcf -c fcfc.conf -j 0 -q bins.dat --data '+dir+'chi2/LRGcat/sigma'+str(Sigma)+'.dat --dd '+dir+'chi2/LRG2pcf/sigma'+str(Sigma)+'.dd --output '+dir+'chi2/LRG2pcf/sigma'+str(Sigma)+'.dat --force 1')

	### calculate the covariance, residuals and chi2
	res = sim-model
	resTcov = res.dot(cov)
	return resTcov.dot(res)

# chi2 minimise
sigma = Minuit(chi2,Sigma=0.3)
sigma.migrad()  # run optimiser
print(sigma.values) 
time_end=time.time()
print('Creating LRG catalogue costs',time_end-time_start,'s')
print('the best LRG distribution sigma is ',sigma.values)


