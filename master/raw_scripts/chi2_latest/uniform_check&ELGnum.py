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
	hdu = fits.open('E:/Master/obs/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits')[1].data
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
for i in range(4):
    zbins =nz[:,2*i]
    sel2 = (zbins>0.6)&(zbins<1.1)
    chi = np.zeros(len(zbins[sel2]))
    for j,dz in enumerate(zbins[sel2]):
        z = np.linspace(0,dz,201)
        chi[j] = np.trapz(c/100/np.sqrt(Om*(1+z)**3+1-Om),dx = z[1]-z[0])

    nchi    = nz[:,2*i+1][sel2]

    chibins=np.copy(chi)

    nchi_chi2 = nchi**2*chi**2
    chi2  = chi**2
    int_nchi2_chi2 = np.sum((nchi_chi2[:-1]+nchi_chi2[1:])/2*(chibins[1:]-chibins[:-1]),axis=0)
    int_chi2 = np.sum((chi2[:-1]+chi2[1:])/2*(chibins[1:]-chibins[:-1]),axis=0)


    neff = np.sqrt(int_nchi2_chi2/int_chi2)
    print('neff in chunck 2{} is {:.7}'.format(i,neff))

'''
ELG chunk densities are 
chunk             21        22            23          25
all          0.00018605, 0.00020091, 0.00017203, 0.00019391
z0.6_1.1     0.0002918873,0.0003150803,0.000268455,0.0003026845
'''


# calculate the number of LRG selected in the UNIT catalogue
c=299792.458
Om=0.31
V = np.array([299841925.8949059,193225157.88905543])
for i,GC in enumerate(['NGC','SGC']):
    list = np.loadtxt('E:/Master/obs/nbar_eBOSS_LRG_'+GC+'_v7_2.dat')
    sel = (list[:,0]>0.6)&(list[:,0]<1.0)
    print(GC+' have {:2.7} galaxies'.format(np.sum(list[:,-1][sel])))
    print('Volumn is {:2.4} Mpc**3'.format(V[i]))
    print('its number density is {:2.7} gal/(Mpc)**3 \n'.format(np.sum(list[:,-1][sel])/V[i]))
    # calculate neff
    
    '''
    chibins = np.zeros(len(list))

    zbins = list[:,0]
    for j,dz in enumerate(zbins):
        z = np.linspace(0,dz,201)
        chibins[j] = np.trapz(c/100/np.sqrt(Om*(1+z)**3+1-Om),dx = z[1]-z[0])

    nchi    = list[:,3]

    nchi_chi2 = nchi**2*chibins**2
    chi2  = chibins**2
    
    int_nchi2_chi2 = np.sum((nchi_chi2[:-1]+nchi_chi2[1:])/2*(chibins[1:]-chibins[:-1]),axis=0)
    int_chi2 = np.sum((chi2[:-1]+chi2[1:])/2*(chibins[1:]-chibins[:-1]),axis=0)
    '''
    nchi_chi2 = np.sum(list[:,5][sel]*list[:,3][sel]**2)
    chi2  = np.sum(list[:,5][sel])
    neff = np.sqrt(int_nchi2_chi2/int_chi2)
    print('neff in {} is {:.7}'.format(GC,neff))

'''
for all volumns:
NGC have 133517.8 galaxies
Volumn is 2.998e+08 Mpc**3
its number density is 0.0004452941 gal/(Mpc)**3

SGC have 86486.0 galaxies
Volumn is 1.932e+08 Mpc**3
its number density is 0.0004475918 gal/(Mpc)**3 

for z in 0.6_1.0:
NGC have 113861.9 galaxies
Volumn is 2.998e+08 Mpc**3
its number density is 0.0003797398 gal/(Mpc)**3 

neff in NGC is 0.0003026845
SGC have 71431.88 galaxies
Volumn is 1.932e+08 Mpc**3
its number density is 0.0003696821 gal/(Mpc)**3 

neff in SGC is 0.0003026845

'''
