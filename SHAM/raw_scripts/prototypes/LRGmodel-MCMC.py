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
from getdist import plots, MCSamples, loadMCSamples
import getdist
import sys
import pymultinest

# variables
gal      = 'LRG'
GC       = sys.argv[1]
date     = sys.argv[2]
npoints  = 100#int(sys.argv[3])
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
fileroot = 'MCMCout/'+date+'/LRG1scat_'+gal+'_'+GC+'/multinest_'
if os.path.exists('MCMCout/'+date+'/LRG1scat_'+gal+'_'+GC)==False:
    os.makedirs('MCMCout/'+date+'/LRG1scat_'+gal+'_'+GC)

# path
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

# generate nseed Gaussian random number arrays in a list
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(data)).astype('float32') for x in range(nseed)] 
print('the uniform random number dtype is ',uniform_randoms[0].dtype)

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
def sham_tpcf(uniform,sigma_high):
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]] #3.06s
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

#from functools import partial
def chi2(sigma_high):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(sigma_high))))
    print('a second calculation')
    with Pool(processes = nseed) as p:
        xi1_tmp = p.starmap(sham_tpcf,zip(uniform_randoms1,repeat(np.float32(sigma_high))))
    
     # average the result for multiple seeds
    xi0,xi2,xi4 = (np.mean(xi0_tmp,axis=0,dtype='float32')[0]+np.mean(xi1_tmp,axis=0,dtype='float32')[0])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[1]+np.mean(xi1_tmp,axis=0,dtype='float32')[1])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[2]+np.mean(xi1_tmp,axis=0,dtype='float32')[2])/2

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        mocks = hdu[1].data[multipole][binmin:binmax,:]
        covcut = np.cov(mocks).astype('float32')
        OBS   = obscf['col3'].astype('float32')
    if multipole=='quad':
        model = np.append(xi0,xi2)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:]))
        covcut  = np.cov(mocks).astype('float32')
        OBS   = np.append(obscf['col3'],obscf['col4']).astype('float32')  
    if multipole=='hexa':
        model = np.append(xi0,xi2,xi4)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:],hdu[1].data[multipole][binmin+400:binmax+400,:]))
        covcut  = np.cov(mocks).astype('float32')
        OBS   = np.append(obscf['col3'],obscf['col4'],obscf['col5']).astype('float32')

    # calculate the covariance, residuals and chi2
    Nbins = len(model)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    return res.dot(covR.dot(res))

# prior
def prior(cube, ndim, nparams):
    cube[0] = cube[0]*2  # uniform between [0,2]

# loglikelihood = -0.5*chi2    
def loglike(cube, ndim, nparams):
    sigma = cube[0]
    return -0.5*chi2(sigma)   
    
# number of dimensions our problem has
parameters = ["sigma"]
npar = len(parameters)

# run MultiNest & write the parameter's name
pymultinest.run(loglike, prior, npar,n_live_points= npoints, outputfiles_basename=fileroot, resume =True , verbose = True,n_iter_before_update=5,write_output=True)
f=open(fileroot+'.paramnames', 'w')
for param in parameters:
    f.write(param+'\n')
f.close()

fin = time.time()  
print('the total {} in {} SHAM costs {:.6} s \n'.format(gal,GC,fin-init))

# getdist plot
sample = loadMCSamples(fileroot)
plt.rcParams['text.usetex'] = False
g = plots.get_single_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g.settings.title_limit_fontsize = 14
g = plots.get_subplot_plotter()
g.plot_1d(sample, 'sigma',title_limit=1)
g.export('LRG-MCMC_posterior_{}_{}_{}_nseed{}.pdf'.format(gal,GC,mode,nseed*2))
plt.close('all')

# results
print('Results:')
stats = sample.getMargeStats()
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
sigma = np.zeros(npar)
for i in range(npar):
    par = stats.parWithName(parameters[i])
    #best[i] = par.bestfit_sample
    mean[i] = par.mean
    sigma[i] = par.err
    lower[i] = par.limits[0].lower
    upper[i] = par.limits[0].upper
    best[i] = (lower[i] + upper[i]) * 0.5
    print('{0:s}: {1:.5f} + {2:.6f} - {3:.6f}, or {4:.5f} +- {5:.6f}'.format( \
        parameters[i], best[i], upper[i]-best[i], best[i]-lower[i], mean[i], \
        sigma[i]))
'''   
# plot the best-fit
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0]))))

if multipole=='mono':    
    fig,ax =plt.subplots(figsize=(8,6))
    ax.errorbar(s,s**2*obscf['col3'],s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
    ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[0],c='m',alpha=0.5)
    label = ['best fit','obs']
    plt.legend(label,loc=0)
    plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
    plt.xlabel('s (Mpc $h^{-1}$)')
    plt.ylabel('s^2 * $\\xi_0$')
    plt.savefig('LRG-MCMC_cf_mono_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig =plt.figure(figsize=(16,6))
    for col,covbin,k in zip(['col3','col4'],[int(0),int(200)],range(2)):
        ax = plt.subplot2grid((1,2),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('LRG-MCMC_cf_quad_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole == 'hexa':
    fig =plt.figure(figsize=(24,6))
    for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
        ax = plt.subplot2grid((1,3),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('LRG-MCMC_cf_hexa_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()

# plot the galaxy probability distribution and the real galaxy number distribution 
n,bins=np.histogram(V,bins=50,range=(0,1000))
fig =plt.figure(figsize=(16,6))
for uniform in uniform_randoms:
    datav = np.copy(V)   
    rand1 = np.append(mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),mean[0]*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = V[(datav<mean[1])]
    LRGorg = org3[np.argpartition(-datav[(datav<mean[1])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(0,1000))
    
    ax = plt.subplot2grid((1,2),(0,0))
    ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5)
    plt.ylabel('prob. to have 1 galaxy in 1 halo')
	plt.title('{} {} distribution: sigma={:.3}'.format(gal,GC,mean[0]))
	plt.xlabel(var+' (km/s)')
	ax.set_xlim(1000,10)

	ax = plt.subplot2grid((1,2),(0,1))
	ax.plot(bins[:-1],n2,alpha=0.5,lw=0.5)
	ax.plot(bins[:-1],n,alpha=0.5,lw=0.5)
	plt.yscale('log')
	plt.ylabel('galaxy numbers')
	plt.title('{} in {}: sigma={:.3}'.format(gal,GC,mean[0]))
	plt.xlabel(var+' (km/s)')
	ax.set_xlim(1000,10)


plt.savefig('LRG-MCMC_distri_'gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
''' 