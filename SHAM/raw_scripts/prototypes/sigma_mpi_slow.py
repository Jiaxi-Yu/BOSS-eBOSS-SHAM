import matplotlib 
matplotlib.use('agg')
import time
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
from covmatrix import covmatrix
from obs import obs
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from functools import partial


# variables
home      = '/global/cscratch1/sd/jiaxi/master/'
rscale = 'linear' # 'log'
GC  = 'NGC' # 'NGC' 'SGC'
zmin     = 0.6
zmax     = 1.0
Om       = 0.31
gal      = 'LRG'
multipole= 'mono' # 'mono','quad','hexa'
mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
obsname  = 'eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
halofile = home+'catalog/halotest120.fits.gz' #home+'catalog/CatshortV.0029.fits.gz'
z = 0.57
boxsize  = 2500
rmin     = 1
rmax     = 50
nbins    = 49
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = 5468750
autocorr = 1
nseed    = 30

# generate s and mu bins
if rscale=='linear':
	bins=np.linspace(rmin,rmax,nbins+1)

if rscale=='log':
	bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
	print('Note: the covariance matrix should also change to log scale.')

s = (bins[:-1]+bins[1:])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

#covariance matrix and the observation 2pcf calculation
if (rmax-rmin)/nbins!=1:
	warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")

covmatrix(home,mockdir,covfits,GC,rmin,rmax,zmin,zmax,Om,os.path.exists(covfits))
obs(home,GC,obsname,randname,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))

# Read the covariance matrix and 
hdu = fits.open(covfits) # cov([mono,quadru])
cov = hdu[1].data['cov'+multipole]
Nbias = (hdu[1].data[multipole]).shape # Nbins=np.array([Nbins,Nm])
covR  = np.linalg.inv(cov)*(Nbias[1]-Nbias[0]-2)/(Nbias[1]-1)
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[rmin:]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=(RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf***************
print('reading the halo catalogue for creating the galaxy catalogue...')
halo = fits.open(halofile)
data = halo[1].data
halo.close()
datac = np.zeros((len(data['vpeak']),5))
for i,key in enumerate(['X','Y','Z','vz','vpeak']):
    datac[:,i] = np.copy(data[key])
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

chifile = 'chi2-sigma.txt'
f=open(chifile,'w')
f.write('# sigma_high sigma_low vmax  chi2 \n')

def tpcf(seed,par):
    np.random.seed(seed)#(seed)
    datav = np.copy(data['vpeak'])
    
    if par[0]=='LRG':
        ### shuffle and pick the Nth maximum values
        sig = par[1]
        datav*=( 1+np.random.normal(scale=sig,size=len(datav)))
        LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]
        print('LRG used')
    if par[0]== 'ELG':
        sigma_high,v_max,sigma_low = par[1],par[2],par[3]
        datav*=( 1+np.random.normal(scale=sigma_high,size=len(datav)))
        org3  = datac[datav<v_max]
        ### the second scattering, select haloes from the heaviest according to the scattered value
        org3[:,4]*= 1+np.random.normal(scale=sigma_low,size=len(org3))
        LRGscat = org3[np.argpartition(-org3[:,4],LRGnum)[:LRGnum]]
        print('ELG used')
    ### transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
    z_redshift %=boxsize
    ### count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
    ### calculate the 2pcf and the multipoles
    return DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1

def chi2(Sigma):
    # calculate mean monopole in parallel
    pool = Pool(nseed)
    part = partial(tpcf,par=['LRG',0.326)#Sigma)
    mono_tmp = pool.map(part,np.arange(nseed))
    pool.close()
    pool.join()
    
    mono = np.mean(mono_tmp,axis=0)
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    ### use trapz to integrate over mu
    xi0 = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2 = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4 = np.trapz(hexa, dx=1./nmu, axis=-1)
    if multipole=='mono':
        model = xi0
        OBS   = obscf['col2']
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)
   
    ### calculate the covariance, residuals and chi2
    res = OBS-model
    f.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))

# chi2 minimise
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
from iminuit import Minuit
sigma = Minuit(chi2,Sigma=0.3,limit_Sigma=(0,0.7),error_Sigma=0.1,errordef=1)
sigma.migrad(precision=0.001)  # run optimiser
print('the best LRG distribution sigma is ',sigma.values[0])
f.close()
time_end=time.time()
print('chi-square fitting finished, costing ',time_end-time_start,'s')


# plot the best fit result
np.random.seed(47)#(seed)
datav = np.copy(data['vpeak'])
### shuffle and pick the Nth maximum values
datav*=( 1+np.random.normal(scale=sigma.values['Sigma'],size=len(datav)))
LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]
z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
z_redshift %=boxsize
DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
### use trapz to integrate over mu
xi0 = np.trapz(mono, dx=1./nmu, axis=1)
fig,ax =plt.subplots()
ax.errorbar(s,s**2*obscf['col2'],s**2*errbar, marker='^',ecolor='k',ls="none")
ax.plot(s,s**2*xi0,c='m',alpha=0.5)
label = ['$\sigma=$'+str(sigma.values['Sigma'])[:5],'obs']
plt.legend(label,loc=0)
plt.title('correlation function')
plt.xlabel('d_cov (Mpc $h^{-1}$)')
plt.ylabel('d_cov^2 * $\\xi$')
plt.savefig('cf_mono_bestfits_extra.png',bbox_tight=True)
plt.close()

'''
# to see the multipole distribution with the same sigma but different random seeds
pool = Pool(nseed)
part = partial(tpcf,sig=0.326)#Sigma)
mono_tmp = pool.map(part,np.arange(nseed))
pool.close()
pool.join()

mono = np.mean(mono_tmp,axis=0)
xi0 = np.trapz(mono, dx=1./nmu, axis=-1)
fig = plt.figure(figsize = (5, 6))
import matplotlib.gridspec as gridspec 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.zeros(nbins),np.mean(xi0,axis=0)]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].errorbar(s,s**2*(np.mean(xi0,axis=0)-value),s**2*np.std(xi0,axis=0),c='k',lw=0.3,ecolor='k',elinewidth=2,label = 'std')
    for i,mono in enumerate(xi0): 
        ax[k,0].plot(s,s**2*(mono-value),lw=0.2)
        
    ax[k,0].set_xlim(0,50)
    plt.legend(loc=0) 
    ax[k,0].set_xlabel('s (Mpc $h^{-1}$)') 
    if k==0:
        ax[k,0].set_ylim(25,105)
        ax[k,0].set_ylabel('$s^2*\\xi$') 
    else:
        ax[k,0].set_ylim(-2,2)
        ax[k,0].set_ylabel('$s^2*\\xi$ diff')
            
plt.savefig('cf_mono_multiseeds.png',bbox_tight=True) 
plt.close('all')  
                                         
'''




