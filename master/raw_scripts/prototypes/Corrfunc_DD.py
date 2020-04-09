import time
time_start=time.time()
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
#from Corrfunc.theory.DD import DD
#from Corrfunc.utils import convert_3d_counts_to_cf

path =  '/global/cscratch1/sd/jiaxi/master/2PCF/0211_LRG_real/'
org = Table.read(path+'LRGorg.dd',format="ascii.no_header")
org1= Table.read(path+'LRGorg.dat',format="ascii.no_header")
DD_counts = Table.read(path+'LRGorgCorrfunc.dd',format='ascii',data_start=1)
#path2 =  '/global/cscratch1/sd/jiaxi/master/catalog/0211_LRG_real/'
#test = Table.read(path2+'LRGorg.dat',format="ascii",data_start=1)
#X,Y,Z = test['x'],test['y'],test['z']
boxsize = 2500
#nthreads = 32
nbins = 100
bins = np.linspace(0, 200, nbins + 1) 

#autocorr=1
#DD_counts = DD(autocorr, nthreads, bins, X, Y, Z,periodic=True, verbose=True,boxsize=boxsize)
#ascii.write(DD_counts,path+'LRGorgCorrfunc.dd')
RR_counts = np.zeros(nbins)
for b in range(nbins):
	RR_counts[b] = 4*np.pi/3*(bins[b+1]**3-bins[b]**3)/(boxsize**3)

xi = DD_counts['npairs']/(5468750**2)/RR_counts-1

#fig,ax=plt.subplots()
#ax.plot(org1['col1'][1:],(org['col3']-DD_counts['npairs'])[1:],c='b')
#plt.title('pair counts difference (org-Corrfunc)')
#plt.xlabel('d_cov (Mpc $h^{-1}$)')
#plt.ylabel('pair counts difference')
#plt.savefig('CorrfunDiff.png',bbox_tight=True)

# 0209
fig,ax=plt.subplots()
x=(DD_counts['rmin']+DD_counts['rmax'])/2
ax.plot(org1['col1'],org1['col1']**2*org1['col2'],c='b',alpha=0.5,label='original')
ax.plot(org1['col1'],org1['col1']**2*xi,c='r',alpha=0.5,label='CorrFunc')
plt.legend(loc=0)
plt.title('correlation function monopole')
plt.xlabel('d_cov (Mpc $h^{-1}$)')
plt.ylabel('d_cov^2 * $\\xi$')
plt.savefig('Corrfun_new.png',bbox_tight=True)

time_start=time.time()
test['x','y','z'].write('galaxy.dat',format='ascii',overwrite=True)
time_end=time.time()
print('saving a galaxy catalogue costs',time_end-time_start,'s')


