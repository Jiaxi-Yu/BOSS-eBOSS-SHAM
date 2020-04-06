import matplotlib 
matplotlib.use('agg')
import time
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
from iminuit import Minuit
import os
from covmatrix import covmatrix
from obs import obs
import warnings
import matplotlib.pyplot as plt


# variables
home      = '/global/cscratch1/sd/jiaxi/master/'
rscale = 'linear' # 'log'
GC  = 'NGC' # 'NGC' 'SGC'
zmin     = 0.6
zmax     = 1.0
Om       = 0.31
multipole= 'quad' # 'mono','quad','hexa'
mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
#*********
covfits = home+'2PCF/obs/corrcoef'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
#********
obsname  = 'eBOSS_LRG_clustering_'+GC+'_v7_2.dat.fits'
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/LRG_'+GC+'.dat'
halofile = home+'catalog/halotest120.fits.gz' #home+'catalog/CatshortV.0029.fits.gz'
z = 0.57
boxsize  = 2500
rmin     = 0
rmax     = 50
nbins    = 50
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = 5468750
autocorr =1

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
errbar = np.std(hdu[1].data[multipole],axis=0)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')  # obs 2pcf
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
datac = np.zeros((len(data['vpeak']),4))
for i,key in enumerate(['X','Y','Z','vz']):
    datac[:,i] = np.copy(data[key])
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# calculate the 2pcf of the galaxies
num = 10
chi2 = np.zeros(num)
sigma = np.linspace(0,1.,num)
xi0,xi2,xi4=[x for x in chi2],[x for x in chi2],[x for x in chi2]
time_start=time.time()
### create the LRG catalogues******************
for i,Sigma in enumerate(sigma):
	#for i in range(num):
	np.random.seed(47)
	#time_start=time.time()
	datav = np.copy(data['vpeak'])
	#datav*=( 1+np.random.normal(scale=0.3,size=len(datav)))
	datav*=( 1+np.random.normal(scale=Sigma,size=len(datav)))
	LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]  # bottleneck.argpartition and np.argpartition ~the same time, np a little bit faster
	#time_end=time.time()
	#print('Sorting the catalogue costs',time_end-time_start,'s')
	### convert to the redshift space
	z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
	z_redshift %=boxsize
	### count the galaxy pairs and normalise them
	DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
	DD_counts['npairs'][0] -=LRGnum
	### calculate the 2pcf and the multipoles
	mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
	quad = mono * 2.5 * (3 * mu**2 - 1)
	hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
	### use trapz to integrate over mu
	xi0[i] = np.trapz(mono, dx=1./nmu, axis=1)
	xi2[i] = np.trapz(quad, dx=1./nmu, axis=1)
	xi4[i] = np.trapz(hexa, dx=1./nmu, axis=1)
	if multipole=='mono':
		model = xi0[i]
		OBS   = obscf['col2']
	elif multipole=='quad':
		model = np.append(xi0[i],xi2[i])
		OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
	else:
		model = np.append(xi0[i],xi2[i],xi4[i])
	### calculate the covariance, residuals and chi2
	res = OBS-model
	resTcovR = res.dot(covR)
	chi2[i]= resTcovR.dot(res)

time_end=time.time()
print('Creating LRG catalogue costs',time_end-time_start,'s')

# chi2 trend:
fig,ax =plt.subplots()
ax.plot(sigma,chi2)
plt.title('#$\chi^2$ test')
plt.xlabel('$\sigma$')
plt.ylabel('$\chi^2$')
plt.savefig('chi2_'+multipole+'.png',bbox_tight=True)
plt.close()

# see the difference between the 2pcf and Corrfunc
for arr,i,col,name in zip([xi0,xi2],range(2),['col2','col3'],['mono','quadru']):
    fig,ax =plt.subplots()
    ax.fill_between(s,s**2*(obscf[col]-errbar[int(50*i):int(50*(i+1))]),s**2*(obscf[col]+errbar[int(50*i):int(50*(i+1))]),color='green', alpha=0.5)
    ax.plot(s,s**2*obscf[col],c='k',alpha=0.6)#,label='obs')
    ax.plot(s,s**2*arr[0],c='r')#,label='$\sigma=0$')
    ax.plot(s,s**2*arr[2],c='c',alpha=0.5)#,label='$\sigma=0.4$')
    ax.plot(s,s**2*arr[4],c='g',alpha=0.5)#,label='$\sigma=0.4$')
    ax.plot(s,s**2*arr[6],c='orange',alpha=0.7)#,label='$\sigma=1.8$')
    ax.plot(s,s**2*arr[8],c='m',alpha=0.5)
    label = ['obs','$\sigma=0$','$\sigma=0.2$','$\sigma=0.4$','$\sigma=0.6$','$\sigma=0.8$']
    plt.legend(label,loc=0)
    plt.title('correlation function: '+name)
    plt.xlabel('d_cov (Mpc $h^{-1}$)')
    plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_'+name+'_largeerr.png',bbox_tight=True)
    plt.close()


# covariance matrix 
fig = plt.figure(figsize=(18,9))
pole=['mono','quad','hexa']
for i,pole in enumerate(pole):
	covfits = home+'2PCF/obs/corrcoef'+pole+'.fits.gz'  
	ax  = plt.subplot2grid((1,3),(0,i))
	hdu = fits.open(covfits) # cov([mono,quadru])
	cov = hdu[1].data['cov'+pole]
	plt.imshow(cov,cmap='Reds')
	plt.title('preproc relative difference ')
	plt.colorbar()


