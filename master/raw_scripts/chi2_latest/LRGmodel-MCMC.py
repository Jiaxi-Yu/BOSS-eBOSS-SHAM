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
from iminuit import Minuit
from getdist import plots, MCSamples, loadMCSamples
import getdist
import sys
import pymultinest


# variables
rscale = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
gal      = 'LRG' 
var      = 'Vpeak'
GC = sys.argv[1]
date  = sys.argv[2]
mode     = sys.argv[3]
npoints  = 100#int(sys.argv[3])
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = int(6.26e4)
autocorr = 1
nseed    = 15
home      = '/global/cscratch1/sd/jiaxi/master/'
fileroot = 'MCMCout/'+date+'/'+gal+'_'+GC+'_'+mode+'/'+gal+'_'+GC+'_'+mode+'_'
if os.path.exists('MCMCout/'+date+'/'+gal+'_'+GC+'_'+mode)==False:
    os.makedirs('MCMCout/'+date+'/'+gal+'_'+GC+'_'+mode)
    
    
zmin     = 0.6
zmax     = 1.0
z = 0.7018
#precut   = 160

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

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=(RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf***************
print('reading the halo catalogue for creating the galaxy catalogue...')
ini = time.time()
halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 
halo = fits.open(halofile)

#sel=(halo[1].data[var]>precut)
halo[1].data = halo[1].data#[sel]
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

end = time.time()
print('{:.5} s'.format(end-ini))

## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate random number arrays once and for all
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)]  
print('the uniform random number dtype is ',uniform_randoms[0].dtype)

# Read the covariance matrix and observation
mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma):
    #datav = np.copy(data[var])
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    if gal=='LRG':
        ### shuffle and pick the Nth maximum values
        rand = np.append(sigma*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
        datav*=( 1+rand)
        LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]

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
    
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(Sigma)))
    
    # average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(xi0_tmp,axis=0)[2]
    '''
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(Sigma))))
    print('the second calculation')
    with Pool(processes = nseed) as p:
        xi1_tmp = p.starmap(sham_tpcf,zip(uniform_randoms1,repeat(np.float32(Sigma))))
    
    # average the result for multiple seeds
    xi0,xi2,xi4 = (np.mean(xi0_tmp,axis=0,dtype='float32')[0]+np.mean(xi1_tmp,axis=0,dtype='float32')[0])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[1]+np.mean(xi1_tmp,axis=0,dtype='float32')[1])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[2]+np.mean(xi1_tmp,axis=0,dtype='float32')[2])/2
    #xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0,dtype='float32')[0],np.mean(xi0_tmp,axis=0,dtype='float32')[1],np.mean(xi0_tmp,axis=0,dtype='float32')[2]
    '''
    
    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        mocks = hdu[1].data[multipole][binmin:binmax,:]
        covcut = np.cov(mocks)
        OBS   = obscf['col3']
    if multipole=='quad':
        model = np.append(xi0,xi2)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:]))
        covcut  = np.cov(mocks)
        OBS   = np.append(obscf['col3'],obscf['col4'])  # obs([mono,quadru])
    if multipole=='hexa':
        model = np.append(xi0,xi2,xi4)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:],hdu[1].data[multipole][binmin+400:binmax+400,:]))
        covcut  = np.cov(mocks)
        OBS   = np.append(obscf['col3'],obscf['col4'],obscf['col5'])
    
    ### calculate the covariance, residuals and chi2
    Nbins = len(model)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    
    return res.dot(covR.dot(res))

# prior
def prior(cube, ndim, nparams):
    cube[0] = cube[0]  # uniform between [0,1]
    #cube[1] = 900*cube[1]+100  # uniform between [0,1000]

# loglikelihood = -0.5*chi2
def loglike(cube, ndim, nparams):
    return -0.5*chi2(cube[0])   
    
# number of dimensions our problem has
parameters = ["sigma"]
n_params = len(parameters)

# run MultiNest & write the parameter's name
pymultinest.run(loglike, prior, n_params,n_live_points= npoints, outputfiles_basename=fileroot, resume =True , verbose = True,n_iter_before_update=5,write_output=True)#,init_MPI =True)
f=open(fileroot+'.paramnames', 'w')
for param in parameters:
    f.write(param+'\n')
f.close()# save parameter names

# getdist plot
sample = loadMCSamples(fileroot)
plt.rcParams['text.usetex'] = False
g = plots.get_single_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g.settings.title_limit_fontsize = 14
g = plots.get_subplot_plotter()
g.plot_1d(sample, 'sigma',title_limit=1)
g.export('{}_{}_'+mode+'_posterior.pdf'.format(gal,GC))
plt.close('all')

print('Results:')
stats = sample.getMargeStats()
npar=1
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
sigma = np.zeros(npar)
for i in range(npar):
    par = stats.parWithName(parameters[i])
#  best[i] = par.bestfit_sample
    mean[i] = par.mean
    sigma[i] = par.err
    lower[i] = par.limits[0].lower
    upper[i] = par.limits[0].upper
    best[i] = (lower[i] + upper[i]) * 0.5
    print('{0:s}: {1:.5f} + {2:.6f} - {3:.6f}, or {4:.5f} +- {5:.6f}'.format( \
        parameters[i], best[i], upper[i]-best[i], best[i]-lower[i], mean[i], \
        sigma[i]))

fin = time.time()  
print('the total {} in {} SHAM costs {:.6} s \n'.format(gal,GC,fin-init))

