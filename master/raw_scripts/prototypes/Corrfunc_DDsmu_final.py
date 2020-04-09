import time
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from Corrfunc.theory import DDsmu
from Corrfunc.theory.DD import DD
import os

# this is a test for LRG model minimisation: so test on catalogues
# use FCFC_box fcfc.conf
path =  '/global/cscratch1/sd/jiaxi/master/catalog/0211_LRG_real/'
lrgcat = Table.read(path+'LRGorg.dat',format="ascii",data_start=1)
X,Y,Z = lrgcat['x'],lrgcat['y'],lrgcat['z']
boxsize = 2500
nthreads = 64
nbins = 100
rmax =200
bins = np.linspace(0, rmax, nbins + 1) 
nmu =120
mu_max = 1
LRGnum=len(X)

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=RR_counts.reshape(nbins,1)+np.zeros((1,nmu))/nmu


time_start=time.time()
# DD pair counting
autocorr=1
DD_smu = DDsmu(autocorr, nthreads, bins,mu_max, nmu, X, Y, Z,periodic=True, verbose=True,boxsize=boxsize)
#DD_smu['npairs'][0] -=LRGnum
DD_s = DD(autocorr, nthreads, bins, X, Y, Z,periodic=True, verbose=True,boxsize=boxsize)
#test the pair counts
orgdd = Table.read('/global/cscratch1/sd/jiaxi/master/2PCF/0211_LRG_real/LRGorg.dd',format="ascii.no_header")
fig,ax=plt.subplots()
ax.plot(orgdd['col1']+2,orgdd['col3'],label='org')
ax.plot(orgdd['col1'],np.sum(DD_smu['npairs'].reshape(nbins,nmu),axis=1),label='ddsmu')
ax.plot(orgdd['col1']-2,DD_s['npairs'],label='dd')
plt.legend(loc=0)
plt.title('pair counts difference (org-Corrfunc)')
plt.xlabel('d_cov (Mpc $h^{-1}$)')
#plt.yscale('log')
plt.ylabel('pair counts difference')
plt.savefig('ddDiff.png',bbox_tight=True)
plt.close()


# calculate the 2pcf
mono = DD_smu['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
xi0 = np.trapz(mono, dx=1./nmu, axis=1)
xi = DD_s['npairs']/(LRGnum**2)/RR_counts.reshape(nbins)-1
# test the monopole results
org = Table.read('/global/cscratch1/sd/jiaxi/master/2PCF/0211_LRG_real/LRGorg.dat',format="ascii.no_header")
fig,ax=plt.subplots()
ax.plot(org['col1'],org['col1']**2*org['col2'],label='org')
ax.plot(bins[:-1],bins[:-1]**2*xi0,alpha=0.5,label='ddsmu')
ax.plot(bins[:-1],bins[:-1]**2*xi,alpha=0.5,label='dd')
plt.legend(loc=0)
plt.title('pair counts difference (org-Corrfunc)')
plt.xlabel('d_cov (Mpc $h^{-1}$)')
#plt.yscale('log')
plt.ylabel('pair counts difference')
plt.savefig('monoDiff_all.png',bbox_tight=True)
plt.close()
##################
#######
mu = DD_counts['mu_max']
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
# use trapz to integrate over mu
xi0 = np.trapz(mono.reshape(nbins,nmu), dx=1./nmu, axis=1)
xi2 = np.trapz(quad.reshape(nbins,nmu), dx=1./nmu, axis=1)
xi4 = np.trapz(hexa.reshape(nbins,nmu), dx=1./nmu, axis=1)
DD = Table([xi0,xi2,xi4])
ascii.write(DD,'LRGCorr.dd',overwrite=True) 
time_end=time.time()
print('Corrfunc costs',time_end-time_start,'s')


# see the difference between the 2pcf and Corrfunc
for arr,i,col,name in zip([xi0,xi2,xi4],range(3),['col2','col3','col4'],['mono','quadru','hexa']):
	fig,ax =plt.subplots()
	ax.plot(org['col1'],org['col1']**2*org[col],c='b',alpha=0.5,label='2pcf')
	ax.plot(bins[:-1],bins[:-1]**2*arr,c='r',alpha=0.5,label='Corrfunc')
	plt.legend(loc=0)
	plt.title('correlation function: '+name)
	plt.xlabel('d_cov (Mpc $h^{-1}$)')
	plt.ylabel('d_cov^2 * $\\xi$')
	plt.savefig(name+'.png',bbox_tight=True)
	plt.close()

# how much time for saving a catalogue
path2 =  '/global/cscratch1/sd/jiaxi/master/catalog/0211_LRG_real/'
test = Table.read(path2+'LRGorg.dat',format="ascii",data_start=1) 
time_start=time.time()
test['x','y','z'].write('galaxy.dat',format='ascii',overwrite=True)
a=Table.read('galaxy.dat',format='ascii.no_header')
#os.system('time ./FCFC_box/2pcf -c ./FCFC/fcfc.copy.conf -d ./galaxy.dat --dd ./galaxy.dd -o galaxy.dat')
time_end=time.time()
print('2pcf calculation costs',time_end-time_start,'s')




