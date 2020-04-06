# calculate the covariance of the mock 2pcf 
# save both the 1000 mock 2pcf and its covariance
import time
time_start=time.time()
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob

# variables(for functions)
home = '/global/cscratch1/sd/jiaxi/master/'
rmin=0
rmax=200
nbins=200
GC = 'NGC'    ## 'NGC' 'SGC'

# read dd , dr and rr files
ddpath = glob.glob('/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/*LRG_'+GC+'*.dd')
nbins=200
nmu=120

# read all the 2pcf data
mockmono = [x for x in range(len(ddpath))]
mockquadru = [x for x in range(len(ddpath))]
mockhexadeca = [x for x in range(len(ddpath))]
# calculate 2pcf multipoles and to form a nbins*nfiles matrix
for i in range(len(ddpath)):
	# read pair-counts files:
	dd = ascii.read(ddpath[i],format='no_header')
	dr = ascii.read(ddpath[i][:-2]+'dr',format='no_header')
	rr = ascii.read(ddpath[i][:-2]+'rr',format='no_header')
	# calculate the 2pcf(mu,s)
	mu = (dd['col1']+dd['col2'])/2
	mono = (dd['col6']-2*dr['col6']+rr['col6'])/rr['col6']
	quad = mono * 2.5 * (3 * mu**2 - 1)
	hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
	# integral 2pcf with mu to have 2pcf(s)
	xi0 = np.trapz(mono.reshape(nbins,nmu), dx=1./nmu, axis=1)
	xi2 = np.trapz(quad.reshape(nbins,nmu), dx=1./nmu, axis=1)
	xi4 = np.trapz(hexa.reshape(nbins,nmu), dx=1./nmu, axis=1) 
	mockmono[i] = xi0
	mockquadru[i] = xi2
	mockhexadeca[i] = xi4

# calculate the covariance
mockmono  = np.array(mockmono).T
mockquadru= np.array(mockquadru).T
mockhexadeca= np.array(mockhexadeca).T
covmono   = np.cov(mockmono)
covquadru = np.cov(mockquadru)
covhexadeca = np.cov(mockhexadeca)

# save data as binary table
# name of the mock 2pcf and covariance matrix file(function return)
mockfits = home+'2PCF/mockcov_'+GC+'.fits.gz'
if os.path.exists(mockfits):
	os.remove(mockfits) 
covfmt = str(nbins)+'D'
mockfmt= str(len(ddpath))+'D'
cols = []
cols.append(fits.Column(name='mono',format=mockfmt, array=mockmono))
cols.append(fits.Column(name='quadru',format=mockfmt, array=mockquadru))
cols.append(fits.Column(name='hexa',format=mockfmt, array=mockhexadeca))
cols.append(fits.Column(name='covmono',format=covfmt, array=covmono))
cols.append(fits.Column(name='covquadru',format=covfmt, array=covquadru))
cols.append(fits.Column(name='covhexa',format=covfmt, array=covhexadeca))
hdulist = fits.BinTableHDU.from_columns(cols)
hdulist.writeto(mockfits, overwrite=True)

time_end=time.time()
print('Covariance matrix calculation costs',time_end-time_start,'s')

