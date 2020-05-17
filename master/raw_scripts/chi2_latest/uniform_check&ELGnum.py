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


c=299792.458
Om=0.31
z1 = np.linspace(0,1.1,201)
z2 = np.linspace(0,0.6,201)
dfin = np.trapz(c/100/np.sqrt(Om*(1+z1)**3+1-Om),dx = z1[1]-z1[0])
dint = np.trapz(c/100/np.sqrt(Om*(1+z2)**3+1-Om),dx = z2[1]-z2[0])
allsky = 4*np.pi*(180/np.pi)**2
V = 4/3*np.pi*(dfin**3-dint**3)*areadeg/allsky

for i,GC in enumerate(['SGC','NGC']):
	hdu = fits.open('/media/jiaxi/disk/Master/obs/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits')[1].data
	galnum=np.sum(hdu['WEIGHT_SYSTOT']*hdu['WEIGHT_CP']*hdu['WEIGHT_NOZ'])# no *hdu['WEIGHT_FKP']
	print(GC+' have {:2.7} galaxies'.format(galnum))
	print('Volumn is {:2.4} Mpc**3'.format(V[i]))
	print('its number density is {:2.7} gal/(Mpc)**3 \n'.format(galnum/V[i]))

## results:
'''
SGC have 102464.6 galaxies
Volumn is 4.789e+08 Mpc**3
its number density is 0.0002139777 gal/(Mpc)**3 

NGC have 98595.65 galaxies
Volumn is 4.237e+08 Mpc**3
its number density is 0.000232783 gal/(Mpc)**3 

their number density difference is around 9%

UNIT ELG numbers
SGC:349256
NGC:410223

finally use 4e5 for NGC/SGC LRG/ELG
'''


# calculate neff
c=299792.458
Om=0.31
nz = np.loadtxt('/media/jiaxi/disk/Master/obs/nbar_eBOSS_ELG_v7.dat')
chi = np.zeros(len(nz)*4)

zbins = np.hstack((nz[:,0],nz[:,2],nz[:,4],nz[:,6]))
for j,dz in enumerate(zbins):
    z = np.linspace(0,dz,201)
    chi[j] = np.trapz(c/100/np.sqrt(Om*(1+z)**3+1-Om),dx = z[1]-z[0])

chibins = np.vstack((chi[:len(nz)],chi[len(nz):len(nz)*2],chi[len(nz)*2:len(nz)*3],chi[len(nz)*3:len(nz)*4])).T
nchi    = np.vstack((nz[:,1],nz[:,3],nz[:,5],nz[:,7])).T

nchi_chi2 = nchi**2*chibins**2
chi2  = chibins**2
int_nchi2_chi2 = np.sum((nchi_chi2[:-1,:]+nchi_chi2[1:,:])/2*(chibins[1:,:]-chibins[:-1,:]),axis=0)
int_chi2 = np.sum((chi2[:-1,:]+chi2[1:,:])/2*(chibins[1:,:]-chibins[:-1,:]),axis=0)


neff = np.sqrt(int_nchi2_chi2/int_chi2)

'''
ELG chunk densities are 
chunk 21        22            23          25
0.00018605, 0.00020091, 0.00017203, 0.00019391
'''

