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

# variables
home      = '/global/cscratch1/sd/jiaxi/master/'
rscale = 'linear' # 'log'
GC  = 'NGC' # 'NGC' 'SGC'
gal = 'ELG'
multipole= 'mono' # 'mono','quad','hexa'
zmin     = 0.6
zmax     = 1.1
Om       = 0.31
z = 0.8594
boxsize  = 1000
rmin     = 5
rmax     = 50
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = 0
autocorr = 1
nseed    = 30

mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
#********
obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
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

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=(RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu
print('the analytical random pair counts are ready.')


#covariance matrix and the observation 2pcf calculation
if (rmax-rmin)/nbins!=1:
	warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")

covmatrix(home,mockdir,covfits,gal,GC,rmin,rmax,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))

# Read the covariance matrix 
hdu = fits.open(covfits) # cov([mono,quadru])
cov = hdu[1].data['cov'+multipole]
Nbias = (hdu[1].data[multipole]).shape # Nbins=np.array([Nbins,Nm])
covR  = np.linalg.inv(cov)*(Nbias[1]-Nbias[0]-2)/(Nbias[1]-1)
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()

# ELG observation:
#*******************************
obs = ascii.read(obs2pcf,format = 'no_header')
obs0,obs1,obs2 = obs['col0'],obs['col1'],obs['col2']

#***********************************
print('the covariance matrix and the observation 2pcf vector are ready.')


# create the halo catalogue and plot their 2pcf***************
print('reading the halo catalogue for creating the galaxy catalogue...')
halo = fits.open(halofile)
data = halo[1].data
halo.close()
for i,key in enumerate(['X','Y','Z','VZ','Vpeak']):
    datac[:,i] = np.copy(data[key])

if len(data['Vpeak'])%2==1:
	datac = datac[:-1,:]

## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# better ways being tested in sigma_mpi
'''
chifile = 'chi2-triple_2.txt'
f=open(chifile,'w')
f.write('# sigma_high sigma_low vmax  chi2 \n')
def chi2(sigma_high,sigma_low,v_max):#(par):#
    np.random.seed(47)#(seed)
    datav = np.copy(data['vpeak'])
    ## shuffle and pick the Nth maximum values
    ### the first scattering + remove heavy halo candidates
    datav*=( 1+np.random.normal(scale=sigma_high,size=len(datav)))
    org3  = datac[datav<v_max]
    ### the second scattering, select haloes from the heaviest according to the scattered value
    org3[:,4]*= 1+np.random.normal(scale=sigma_low,size=len(org3))
    LRGscat = org3[np.argpartition(-org3[:,4],LRGnum)[:LRGnum]]
    ## transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
    z_redshift %=boxsize
    ## count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
    ## calculate the 2pcf and the multipoles
    mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    ## use trapz to integrate over mu
    xi0 = np.trapz(mono, dx=1./nmu, axis=1)
    xi2 = np.trapz(quad, dx=1./nmu, axis=1)
    xi4 = np.trapz(hexa, dx=1./nmu, axis=1)
    if multipole=='mono':
        model = xi0#.mean(axis=-1)
        OBS   = obs0
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        OBS   = np.append(obs0,obs1)  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)
	OBS   = np.appen(obs0,obs1,obs2)
   
    ### calculate the covariance, residuals and chi2
    res = OBS-model
    f.write('{} {} {} {} \n'.format(sigma_high,sigma_low,v_max,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))
    #return model
'''
# chi2 minimise
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
from iminuit import Minuit
sigma = Minuit(chi2,sigma_high=0.3,sigma_low=0.336,v_max=100,limit_sigma_high=(0,2),limit_sigma_low=(0,2),limit_v_max=(0,500),errordef=1) 
# 3 parameters: 1 or 2 param needed to be fix to see the trend
# fix_v_max=True,error_sigma_high = 0.01,error_sigma_low = 0.05,error_v_max = 100,
sigma.migrad(precision=0.01)  # run optimiser
print('the best LRG distribution sigma is ',sigma.values)
print('chi-square fitting finished.')
time_end=time.time()
print('Creating LRG catalogue costs',time_end-time_start,'s')
f.close()

names = ['sigma_high','sigma_low','v_max']
a=np.loadtxt(chifile,unpack=True)
# plot the chi2-sigma relation(single parameters)
for k,name in enumerate(names):
    plt.scatter(a[k],a[-1])
    plt.xlabel(name)
    plt.ylabel('$\chi^2$')
    plt.savefig(chifile[:-4]+'_'+name+'.png',bbox_tight=True)
    plt.close()


# plot the best fit result
np.random.seed(47)#(seed)
datav = np.copy(data['vpeak'])
## shuffle and pick the Nth maximum values
### the first scattering + remove heavy halo candidates
datav*=( 1+np.random.normal(scale=sigma.values[0],size=len(datav)))
org3  = datac[datav<sigma.values[2]]
### the second scattering, select haloes from the heaviest according to the scattered value
org3[:,4]*= 1+np.random.normal(scale=sigma.values[1],size=len(org3))
LRGscat = org3[np.argpartition(-org3[:,4],LRGnum)[:LRGnum]]
## transfer to the redshift space
z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
z_redshift %=boxsize
## count the galaxy pairs and normalise them
DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
## calculate the 2pcf and the multipoles
mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
quad = mono * 2.5 * (3 * mu**2 - 1)
hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
## use trapz to integrate over mu
xi0 = np.trapz(mono, dx=1./nmu, axis=1)
fig,ax =plt.subplots()
ax.errorbar(s,s**2*obscf['col2']*0.5,s**2*errbar, marker='s',ecolor='k',ls="none")
ax.plot(s,s**2*xi0,c='m',alpha=0.5)
label = ['bestfit','obs']
plt.legend(label,loc=0)
plt.title('$\sigma_{high}=$'+str(sigma.values[0])[:5]+', $v_{max}=$'+str(sigma.values[2])[:3]+' km/s, $\sigma_{low}=$'+str(sigma.values[1])[:5])
plt.xlabel('d_cov (Mpc $h^{-1}$)')
plt.ylabel('d_cov^2 * $\\xi$')
plt.savefig('cf_mono_bestfits_ELG2.png',bbox_tight=True)
plt.close()


