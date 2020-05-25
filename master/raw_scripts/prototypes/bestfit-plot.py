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
import matplotlib.gridspec as gridspec 
from multiprocessing import Pool 
from itertools import repeat
import glob
import sys

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
id   = int(sys.argv[3])
mean = [x for x in range(2)]
mean[0] = np.float(sys.argv[4])
mean[1] = np.float(sys.argv[5])
func     = ['HAM','LRG1scat']
nseed    = 15
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

# covariance matrix and the observation 2pcf path
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
    obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
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
if rscale=='log':
    nbins = 50
    bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    print('Note: the covariance matrix should also change to log scale.')
if (rmax-rmin)/nbins!=1:
    warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")
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
datac = np.zeros((len(data),4))
for i,key in enumerate(['X','Y','Z','VZ']):
    datac[:,i] = np.copy(data[key])
V = np.copy(data[var]).astype('float32')
datac = datac.astype('float32')
half = int(len(data)/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 

# generate covariance matrices and observations
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'  
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))
# Read the covariance matrices and observations
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:LRGnum]] #3.06s
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
    return [xi0_single,xi2_single,xi4_single]
    
# plot the best-fit
if id==0:
    with Pool(processes = nseed) as p:
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0])),repeat(np.float32(mean[1]))))
if id==1:
    with Pool(processes = nseed) as p:
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0])),repeat(np.float32(1e5))))


if multipole=='mono':    
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    values=[np.zeros(nbins),np.mean(xi_ELG,axis=0)[0]]
    for j in range(2):
        ax[j,0] = fig.add_subplot(spec[j,0])
        ax[j,0].errorbar(s,s**2*(obscf['col3']-values[j]),s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*(np.mean(xi_ELG,axis=0)[0]-values[j]),c='m',alpha=0.5)
        plt.xlabel('s (Mpc $h^{-1}$)')
        if (j==0):
            ax[j,0].set_ylabel('$s^2*\\xi_0$')
            label = ['best fit','obs']
            plt.legend(label,loc=1)
            if id==0:
                plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
            if id==1:
                plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
        if (j==1):
            ax[j,0].set_ylabel('$s^2 \Delta\\xi_0$')

    plt.savefig(func[id]+'-bestfit_cf_mono_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,k in zip(['col3','col4'],[int(0),int(200)],range(2)):
        values=[np.zeros(nbins),np.mean(xi_ELG,axis=0)[k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
            ax[j,k].plot(s,s**2*(np.mean(xi_ELG,axis=0)[k]-values[j]),c='m',alpha=0.5)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                label = ['best fit','obs']
                plt.legend(label,loc=1)
                if id==0:
                    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
                if id==1:
                    plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig(func[id]+'-bestfit_cf_quad_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole == 'hexa':
    fig = plt.figure(figsize=(21,8))
    spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,3), dtype=type(plt.axes))
    for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
        values=[np.zeros(nbins),np.mean(xi_ELG,axis=0)[k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
            axax[j,k].plot(s,s**2*(np.mean(xi_ELG,axis=0)[k]-values[j]),c='m',alpha=0.5)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                label = ['best fit','obs']
                plt.legend(label,loc=1)
                if id==0:
                    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
                if id==1:
                    plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig(func[id]+'-bestfit_cf_hexa_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()

'''
vhigh = [mean[1],1e5]
# plot the galaxy probability distribution and the real galaxy number distribution 
n,bins=np.histogram(V,bins=50,range=(0,1000))
fig =plt.figure(figsize=(16,6))
ax = plt.subplot2grid((1,2),(0,0))
for uniform in uniform_randoms:
    datav = np.copy(V)   
    rand1 = np.append(mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = V[(datav<vhigh[id])]
    LRGorg = org3[np.argpartition(-datav[(datav<vhigh[id])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(0,1000))
    ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5)
plt.ylabel('prob. to have 1 galaxy in 1 halo')
if id==0:
    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
if id==1:
    plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)

ax = plt.subplot2grid((1,2),(0,1))
for uniform in uniform_randoms:
    datav = np.copy(V)   
    rand1 = np.append(mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = V[(datav<vhigh[id])]
    LRGorg = org3[np.argpartition(-datav[(datav<vhigh[id])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(0,1000))
    ax.plot(bins[:-1],n2,alpha=0.5,lw=0.5)
    ax.plot(bins[:-1],n,alpha=0.5,lw=0.5)
plt.yscale('log')
plt.ylabel('galaxy numbers')
if id==0:
    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
if id==1:
    plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)

plt.savefig(func[id]+'-bestfit_distr_'+gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
'''