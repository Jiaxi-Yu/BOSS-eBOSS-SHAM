# calculate the covariance of the mock 2pcf 
# save both the 1000 mock 2pcf and its covariance
# and calculate the 2pcf of observations
import time
time_start=time.time()
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
import os
import glob

# variables(for functions)
def covmatrix(home,mockdir,mockfits,GC,rmin,rmax,zmin,zmax,Om,exist):
	print('new ver')
	if exist ==True:
		print('The mock covariance file already exists.')
	else:
		nbins=200
		nmu=120

		# read dd , dr and rr files
		ddpath = glob.glob(mockdir+'*LRG_'+GC+'*.dd')
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
			s = ((dd['col3']+dd['col4'])/2).reshape(nbins,nmu)[:,0]
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
			if (i%100==0)|(i==999):
				print('LRG_'+GC+'_mock ',np.ceil(i/10),'% 2pcf completed.')
		    
		# calculate the covariance
		mockmono  = np.array(mockmono).T[:rmax]
		mockquadru= np.array(mockquadru).T[:rmax]
		mockhexadeca= np.array(mockhexadeca).T[:rmax]
	    
		covmono   = np.cov(mockmono)
		covquadru = np.cov(np.vstack((mockmono,mockquadru)))
		covhexadeca = np.cov(np.vstack((mockmono,mockquadru,mockhexadeca)))
		print('shape of the covariance matrices for [mono],[mono,quadru] and [mono,quad,hexa]')
		print(covmono.shape,covquadru.shape,covhexadeca.shape)

		# save data as binary table
		# name of the mock 2pcf and covariance matrix file(function return)
		for name,mockarr,cova in zip(['mono','quad','hexa'],[mockmono,mockquadru,mockhexadeca],[covmono,covquadru,covhexadeca]):
			cols = []
			cols.append(fits.Column(name=name,format=str(len(ddpath))+'D',array=mockarr))
			cols.append(fits.Column(name='cov'+name,format=str(len(cova))+'D',array=cova))
			
			hdulist = fits.BinTableHDU.from_columns(cols)
			hdulist.header.update(rmin=rmin,rmax=rmax,sbins=nbins,nmu=nmu)
			hdulist.writeto(mockfits[:51]+name+mockfits[-8:], overwrite=True)

		time_end=time.time()
		print('Covariance matrix calculation costs',time_end-time_start,'s')

