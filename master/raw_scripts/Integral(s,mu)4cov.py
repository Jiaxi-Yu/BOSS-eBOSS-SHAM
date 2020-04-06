# test whether the 2pcf multipoles are the similar to the result of integral
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from Corrfunc.mocks import DDsmu_mocks
from scipy.integrate import simps
import os
import glob

# this is a test on mock for the covariance matrix
# use ./FCFC/2pcf -c fcfc.conf
GC = 'NGC'       ## 'NGC' 'SGC'
num = '0001' #'0039'#'0871' ## '0155' '0906'
# dd files for mocks
ddpath = glob.glob('/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/*LRG_'+GC+'*'+num+'.dd')
drpath = glob.glob('/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/*LRG_'+GC+'*'+num+'.dr')
rrpath = glob.glob('/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z0.6z1.0/2PCF/*LRG_'+GC+'*'+num+'.rr')

# read pair-counts files:
dd = ascii.read(ddpath[0],format='no_header')
dr = ascii.read(drpath[0],format='no_header')
rr = ascii.read(rrpath[0],format='no_header')


# calculate the 2pcf(s,mu)
mu = (dd['col1']+dd['col2'])/2
s = ((dd['col3']+dd['col4'])/2).reshape(nbins,nmu)[:,0]
mono = (dd['col6']-2*dr['col6']+rr['col6'])/rr['col6']
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)

nmu=120
nbins=200
# integrate by mu to get 2pcf(s)
xi0,xi2,xi4 = [0,1],[0,1],[0,1]
#xi0[0] = simps(mono.reshape(nbins,nmu), dx=1./nmu, axis=1,even='last')
#xi2[0] = simps(quad.reshape(nbins,nmu), dx=1./nmu, axis=1,even='last')
#xi4[0] = simps(hexa.reshape(nbins,nmu), dx=1./nmu, axis=1,even='last')
xi0[0] = np.trapz(mono.reshape(nbins,nmu), dx=1./nmu, axis=1)
xi2[0] = np.trapz(quad.reshape(nbins,nmu), dx=1./nmu, axis=1)
xi4[0] = np.trapz(hexa.reshape(nbins,nmu), dx=1./nmu, axis=1) 

# 2pcf processed mocks
home = '/global/cscratch1/sd/jiaxi/master/2PCF/IntegralTest/'
#org = Table.read(home+GC+'_'+num+'.dat',format='ascii.no_header')
#org2 = Table.read(home+'Om0.307/'+GC+'_'+num+'.dat',format='ascii.no_header')
org1 = Table.read(GC+'_'+num+'_1Mpc.dat',format='ascii.no_header')
org5 = Table.read(GC+'_'+num+'_5Mpc.dat',format='ascii.no_header')

# read the 2pcf result

fig,ax =plt.subplots(figsize=(8,5))
for arr,i,col,name in zip([xi0,xi2],range(2),['col2','col3'],['mono','quadru']):
	#ax.plot(org['col1'],org['col1']**2*org[col],c='b',linewidth=1,label ='2pcf_1Mpc')
	#ax.plot(org2['col1'],org2['col1']**2*org2[col],c='k',alpha=0.4,label='2pcf Om0.307',linewidth=1)
	#ax.plot(np.arange(200),np.arange(200)**2*arr[0],c='orange',alpha=0.7,label='simps',linewidth=1)
	#ax.plot(org1['col1'],org1['col1']**2*org1[col],ls='-.',linewidth=1,label = '2pcf_1Mpc')
	ax.plot(org5['col1'],org5['col1']**2*org5[col],label = '2pcf_5Mpc')
	ax.plot(s,s**2*arr[0],ls='--',linewidth=0.5,label='trapz_1Mpc')
	#plt.ylim(-250,110)
	plt.title('correlation function comparison')
	plt.xlabel('d_cov (Mpc $h^{-1}$)')
	plt.ylabel('d_cov^2 * $\\xi$')
	plt.legend(loc=0)


plt.savefig(GC+num+'_5Mpc.png',bbox_tight=True)
plt.close()

# check in detail dd,dr,rr data and see where comes the difference
# for pair counts: name['col5'],org3['col3']\
# for normalised counts: name['col6'],org3['col4']
for name,lab in zip([dd,dr,rr],['dd','dr','rr']):
	org3 = Table.read(home+GC+'_'+num+'.'+lab,format='ascii.no_header')
	fig,ax =plt.subplots(figsize=(10,5))
	ax.plot(org3['col1'],np.sum(name['col5'].reshape(nbins,nmu),axis=1)/org3['col3']-1)#,alpha=0.5,label='paircounts files',linewidth=0.5)
	#ax.plot(org3['col1'],org3['col3'],label='2pcf files',alpha=0.5,linewidth=0.5)
	plt.legend(loc=2)
	plt.title(lab+' relative counts difference ')
	plt.xlabel('d_cov (Mpc $h^{-1}$)')
	plt.ylabel(lab+'_relative counts difference')
	plt.savefig(GC+num+'_'+lab+'_reladiff.png',bbox_tight=True)
	plt.close()

	#table = Table([np.sum(name['col5'].reshape(nbins,nmu),axis=1),org3['col3']],names=['files_'+lab,'2pcf_'+lab])
	#table.write((GC+num+'_'+lab+'_diff.dat'),format='ascii',overwrite=True)



