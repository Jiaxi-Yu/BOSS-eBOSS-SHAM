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


# variables
rscale = 'linear' # 'log'
GC  = 'NGC' # 'NGC' 'SGC'
gal      = 'LRG'  
multipole= 'mono' # 'mono','quad','hexa'

zmin     = 0.6
zmax     = 1.0
Om       = 0.31
z = 0.8594
boxsize  = 1000
rmin     = 5
rmax     = 50
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = 5468750
autocorr = 1
nseed    = 30

home      = '/global/cscratch1/sd/jiaxi/master/'
if gal=='LRG':
	mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
	obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
else:
	mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
	obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'

covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
halofile = home+'catalog/UNIT_hlist_0.53780-cut.fits.gz' 



# generate s and mu bins
if rscale=='linear':
	bins  = np.arange(rmin,rmax+1,1)
	nbins = len(bins)-1

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
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,True)

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
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ','Vpeak']):
    datac[:,i] = np.copy(data[key])

if len(data['Vpeak'])%2==1:
	datac = datac[:-1,:]

## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate random number arrays once and for all
## generate an array of uniform random numbers for a given seed
def generate_random(seed):
	filename = home+'catalog/randnum/random_'+str(seed)+'.fits.gz'
	uni = np.random.RandomState(seed=seed).rand(len(data))
	Table([uni[:int(len(data)/2)],uni[int(len(data)/2):]],dtype=[np.float32,np.float32]).write(filename, format='fits',overwrite=True)
	return Table([uni[:int(len(data)/2)],uni[int(len(data)/2):]],dtype=[np.float32,np.float32])
## read the existing data
def read_random(seed):
	filename = home+'catalog/randnum/random_'+str(seed)+'.fits.gz'
	return fits.open(filename)[1].data

# generate nseed arrays of uniform random numbers    
if os.path.exists(home+'catalog/randnum/')==False:
	os.mkdir(home+'catalog/randnum/')
	print('writing random number files...')
	with Pool(processes = nseed) as p:
		uniform_randoms = p.map(generate_random,np.arange(nseed)+1)
else:
	#ranpath = glob.glob(home+'catalog/randnum/*')
	print('reading random number files...')
	with Pool(processes = nseed) as p:
		uniform_randoms = p.map(read_random,np.arange(nseed)+1)



def sham_tpcf(uniform,sigma):
	datav = np.copy(data['Vpeak'])
    	# shuffle the halo catalogue and select those have a galaxy inside
	if gal=='LRG':
        	### shuffle and pick the Nth maximum values
		rand = np.append(sigma*np.sqrt(-2*np.log(uniform['col0']))*np.cos(2*np.pi*uniform['col1']),sigma*np.sqrt(-2*np.log(uniform['col0']))*np.sin(2*np.pi*uniform['col1'])) 
		datav*=( 1+rand)
		LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]
		print('LRG used')
	if gal== 'ELG':
		sigma_high,v_max,sigma_low = par[1],par[2],par[3]
		rand1 = np.append(sigma*np.sqrt(-2*np.log(uniform['col0']))*np.cos(2*np.pi*uniform['col1']),sigma*np.sqrt(-2*np.log(uniform['col0']))*np.sin(2*np.pi*uniform['col1'])) 
		datav*=( 1+rand1)
		org3  = datac[datav<v_max]
		if len(org3)%2==1:
			org3 = org3[:-1]
        	### the second scattering, select haloes from the heaviest according to the scattered value
		rand2 = np.append(sigma*np.sqrt(-2*np.log(uniform['col0'][:int(len(org3)+1)]))*np.cos(2*np.pi*uniform['col1']),sigma*np.sqrt(-2*np.log(uniform['col0']))*np.sin(2*np.pi*uniform['col1'])) 

		datav*=( 1+rand)
		org3[:,4]*= 1+np.random.normal(scale=sigma_low,size=len(org3))
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

def chi2(Sigma):
    	# calculate mean monopole in parallel
	with Pool(processes = nseed) as p:
        	xi_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(Sigma)))
    
	# average the result for multiple seeds
	xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi2_tmp,axis=0)[1],np.mean(xi4_tmp,axis=0)[2]
    
	# identify the fitting multipoles
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
	#f.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
	return res.dot(covR.dot(res))

# chi2 minimise
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 

sigma = Minuit(chi2,Sigma=0.3,limit_Sigma=(0,0.7),error_Sigma=0.1,errordef=1)
sigma.migrad(precision=0.001)  # run optimiser
print('the best LRG distribution sigma is ',sigma.values[0])
#f.close()
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




