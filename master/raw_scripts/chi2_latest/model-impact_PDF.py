import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
from covmatrix import covmatrix
from obs import obs
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec 
import sys

# variables
gal      = 'ELG'
GC       = 'SGC'
date     = '0526' #'0523' '0523-1'
func = 'HAM'
mean = [x for x in range(2)]
mean[0] = np.float(sys.argv[1])
mean[1] = np.float(sys.argv[2])
id = 0
npoints  = 200
nseed    = 10
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home      = '/global/cscratch1/sd/jiaxi/master/'

if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    precut   = 80
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
    halofile = home+'catalog/UNIT_hlist_0.53780.fits.gz' 
if gal == 'LRG':
    LRGnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    precut   = 160
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS+SEQUELS_'+gal+'_'+GC+'_v7_2.dat'
    halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 
    
# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate separation bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    
# generate mu bins   
s = (bins[:-1]+bins[1:])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf
print('reading the halo catalogue for creating the galaxy catalogue...')
halo = fits.open(halofile)
# make sure len(data) is even
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
halo.close()

print('selecting only the necessary variables...')
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac[:,i] = np.copy(data[key])
datac = datac.astype('float32')
half = int(len(data)/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 

# generate covariance matrices and observations
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'  
#randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))
# Read the covariance matrices and observations
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(datac[:,-1])
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:LRGnum]] #3.06s
    n2,bins2=np.histogram(LRGscat[:,-1],bins=50,range=(0,1000))
    # transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
    z_redshift %=boxsize
    # count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
    # calculate the 2pcf and the multipoles
    mono = (DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1)
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    # use trapz to integrate over mu
    xi0_single = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2_single = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4_single = np.trapz(hexa, dx=1./nmu, axis=-1)
    return [xi0_single,xi2_single,xi4_single,n2]

extra = [np.float(sys.argv[3])]
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0])),repeat(np.float32(mean[1]))))
    
with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0]*(1-extra[id]))),repeat(np.float32(mean[1]))))

with Pool(processes = nseed) as p:
    xi2_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0])),repeat(np.float32(mean[1]*(1+extra[id])))))
    
type=['Vcut','sigma']
fig =plt.figure(figsize=(18,5))
for col,covbin,k,name in zip(['col3','col4'],[int(0),int(200)],range(2),['monopole','quadrupole']):
    ax = plt.subplot2grid((1,3),(0,k)) 
    ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none",label= 'obs')
    ax.plot(s,s**2*np.mean(xi1_ELG,axis=0)[k],c='c',alpha=0.8,label='$\sigma$={:.3},Vcut={}km/s'.format(mean[0]*(1-extra[id]),round(mean[1])))
    ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.7,label='$\sigma$={:.3},Vcut={}km/s' .format(mean[0],round(mean[1])))
    ax.plot(s,s**2*np.mean(xi2_ELG,axis=0)[k],c='b',alpha=0.8,label='$\sigma$={:.3},Vcut={}km/s'.format(mean[0],round(mean[1]*(1+extra[id]))))
    plt.legend(loc=0)
    plt.title('correlation function {}'.format(name))
    plt.xlabel('s(Mpc $h^{-1}$)')
    plt.ylabel('$s^2 * \\xi_{}$'.format(k*2))

    
nlist = [xi_ELG[x][3] for x in range(nseed)]
narray = np.array(nlist).T
n1list = [xi1_ELG[x][3] for x in range(nseed)]
n1array = np.array(n1list).T
n2list = [xi2_ELG[x][3] for x in range(nseed)]
n2array = np.array(n2list).T

# Vpeak PDF
n,bins=np.histogram(datac[:,-1],bins=50,range=(0,1000))
binmid = (bins[:-1]+bins[1:])/2
ax = plt.subplot2grid((1,3),(0,2)) 
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[3]/(n),yerr=np.std(n1array,axis=-1)/(n+1),color='c',alpha=0.8,ecolor='c',ds='steps-mid',label='$\sigma$={:.3},Vcut={} km/s'.format(mean[0]*(1-extra[id]),round(mean[1])))
ax.errorbar(binmid,np.mean(xi_ELG,axis=0)[3]/(n),yerr = np.std(narray,axis=-1)/(n+1),color='m',alpha=0.7,ecolor='m',ds='steps-mid',label='$\sigma$={:.3},Vcut={} km/s'.format(mean[0],round(mean[1])))
ax.errorbar(binmid,np.mean(xi2_ELG,axis=0)[3]/(n),yerr=np.std(n2array,axis=-1)/(n+1),color='b',alpha=0.8,ecolor='b',ds='steps-mid',label='$\sigma$={:.3},Vcut={} km/s'.format(mean[0],round(mean[1]*(1+extra[id]))))
ax.plot(binmid[np.argmax(np.mean(xi1_ELG,axis=0)[3]/(n+1))]*np.ones(5),np.linspace(0,max(np.mean(xi1_ELG,axis=0)[3]/(n+1)),5),'c--',alpha=0.8)
ax.plot(binmid[np.argmax(np.mean(xi_ELG,axis=0)[3]/(n+1))]*np.ones(5),np.linspace(0,max(np.mean(xi_ELG,axis=0)[3]/(n+1)),5),'m--',alpha=0.7)
ax.plot(binmid[np.argmax(np.mean(xi2_ELG,axis=0)[3]/(n+1))]*np.ones(5),np.linspace(0,max(np.mean(xi2_ELG,axis=0)[3]/(n+1)),5),'b--',alpha=0.8)

plt.ylabel('galaxy numbers')
plt.legend(loc=2)
plt.title('Vpeak probability distribution')
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,0)
ax.set_ylim(0,0.055)

plt.savefig('model_impact+PDF_'+gal+'_'+GC+'.png',bbox_tight=True)
plt.close()