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
from itertools import repeat
import glob
from iminuit import Minuit
import sys


# variables
rscale = 'linear' # 'log'
GC  = 'NGC' # 'NGC' 'SGC'
gal      = sys.argv[1] #'LRG'  
multipole= 'mono' # 'mono','quad','hexa'


Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 50
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = int(4e5)
autocorr = 1
nseed    = 30

home      = '/global/cscratch1/sd/jiaxi/master/'
if gal=='LRG':
	zmin     = 0.6
	zmax     = 1.0
	z = 0.8594
	mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
	obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
	halofile = home+'catalog/UNIT_hlist_0.53780-cut.fits.gz' 
else:
	zmin     = 0.6
	zmax     = 1.1
	z = 0.8594
	mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
	obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
	halofile = home+'catalog/UNIT_hlist_0.53780-cut.fits.gz' 

covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'

# generate s and mu bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax

if rscale=='log':
	nbins = 50
	bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
	print('Note: the covariance matrix should also change to log scale.')

s = (bins[:-1]+bins[1:])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

#covariance matrix and the observation 2pcf calculation
if (rmax-rmin)/nbins!=1:
	warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")

covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))

# Read the covariance matrix and 
hdu = fits.open(covfits) # cov([mono,quadru])
cov = hdu[1].data['cov'+multipole]
Nbins,Nmock = (hdu[1].data[multipole]).shape # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)[binmin:binmax]
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=(RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf***************
print('reading the halo catalogue for creating the galaxy catalogue...')
halo = fits.open(halofile)
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
    
halo.close()
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ','Vpeak']):
    datac[:,i] = np.copy(data[key])

half = int(len(data)/2)
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate random number arrays once and for all
uniform_randoms = [np.random.RandomState(seed=int(x+1)).rand(len(data)) for x in range(nseed)]
print('uniform random number arrays are ready.')

# HAM application
def sham_tpcf(uniform,sigma):
    datav = np.copy(data['Vpeak'])
    	# shuffle the halo catalogue and select those have a galaxy inside

    if gal=='LRG':
        ### shuffle and pick the Nth maximum values
        rand = np.append(sigma*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
        datav*=( 1+rand)
        LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]
        print('LRG used')

    if gal== 'ELG':
        sigma_high,v_max,sigma_low = sigma[0],sigma[1],par[2]
        rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
        datav*=( 1+rand1)
        org3  = datac[datav<v_max]
        if len(org3)%2==1:
            org3 = org3[:-1]		
        length = int(len(org3)/2)
        ### the second scattering, select haloes from the heaviest according to the scattered value
        rand2 = np.append(sigma_low*np.sqrt(-2*np.log(uniform[:length]))*np.cos(2*np.pi*uniform[-length:]),sigma_low*np.sqrt(-2*np.log(uniform[:length]))*np.sin(2*np.pi*uniform[-length:])) 
        org3[:,4]*=( 1+rand2)
        LRGscat = org3[np.argpartition(-org3[:,4],LRGnum)[:LRGnum]]
        print('ELG used')

    ### transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
    z_redshift %=boxsize
     ### count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
        ### calculate the 2pcf and the multipoles
    mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    ### use trapz to integrate over mu
    xi0_single = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2_single = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4_single = np.trapz(hexa, dx=1./nmu, axis=-1)
    return [xi0_single,xi2_single,xi4_single]

#from functools import partial
def chi2(Sigma):
# calculate mean monopole in parallel
    print('parallel calculation')
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(Sigma)))
    
	# average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(xi0_tmp,axis=0)[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        covcut = cov[binmin:binmax,binmin:binmax]
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = obscf['col2']
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+binmax:binmax+binmax,:]))
        covcut  = np.hstack((cov[:,binmin:binmax],cov_tmp[:,binmin+binmax:binmax+binmax]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)

        ### calculate the covariance, residuals and chi2
    res = OBS-model
    #f.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))

#from functools import partial
xi0_tmp=[x for x in np.arange(nseed)]
def chi2_slow(Sigma):
# calculate mean monopole in parallel
    print('loop calculation')
    for i,uni in enumerate(uniform_randoms):
        xi0_tmp[i] = sham_tpcf(uni,Sigma)
    
	# average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(xi0_tmp,axis=0)[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        covcut = cov[binmin:binmax,binmin:binmax]
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = obscf['col2']
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+binmax:binmax+binmax,:]))
        covcut  = np.hstack((cov[:,binmin:binmax],cov_tmp[:,binmin+binmax:binmax+binmax]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)

        ### calculate the covariance, residuals and chi2
    res = OBS-model
    #f.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))
    
# chi2 minimise
chifile = gal+'_chi2_parallel.txt'
f=open(chifile,'w')
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
if gal=='LRG':
    sigma = Minuit(chi2,Sigma=0.3,limit_Sigma=(0,0.7),error_Sigma=0.1,errordef=1)
    sigma.migrad(precision=0.001)  # run optimiser
    time_end=time.time()
    f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
    f.write('parallel calculation best param {:.3} \n'.format(sigma.values[0]))
else:
    sigma = Minuit(chi2,sigma_high=0.3,sigma_low=0.336,v_max=100,limit_sigma_high=(0,2),limit_sigma_low=(0,2),limit_v_max=(0,500),errordef=1) 
    sigma.migrad(precision=0.001)  # run optimiser
    time_end=time.time()
    f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
    f.write('parallel calculation best param sigma_high, sigma_low, v_max = {:.3},{:.3},{:.4} km/s \n'.format(sigma.values[0],sigma.values[1],sigma.values[2]))
#

# chi2 minimise
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
if gal=='LRG':
    sigma = Minuit(chi2_slow,Sigma=0.3,limit_Sigma=(0,0.7),error_Sigma=0.1,errordef=1)
    sigma.migrad(precision=0.001)  # run optimiser
    time_end=time.time()
    f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
    f.write('parallel calculation best param {:.3} \n'.format(sigma.values[0]))
else:
    sigma = Minuit(chi2_slow,sigma_high=0.3,sigma_low=0.336,v_max=100,limit_sigma_high=(0,2),limit_sigma_low=(0,2),limit_v_max=(0,500),errordef=1) 
    sigma.migrad(precision=0.001)  # run optimiser
    time_end=time.time()
    f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
    f.write('parallel calculation best param sigma_high, sigma_low, v_max = {:.3},{:.3},{:.4} km/s \n'.format(sigma.values[0],sigma.values[1],sigma.values[2]))
#

f.close()
'''
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
plt.savefig('cf_mono_bestfits_multiprocessing.png',bbox_tight=True)
plt.close()
'''
'''
# to see the multipole distribution with the same sigma but different random seeds
pool = Pool(nseed)
part = partial(tpcf,sig=0.326)#Sigma)
mono_tmp = pool.map(part,np.arange(nseed))
pool.close()
pool.join()

mono = np.mean(mono_tmp,axis=0)
xi0 = np.trapz(mono, dx=1./nmu, axis=-1)
fig = plt.figure(figsize = (8, 6))
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




