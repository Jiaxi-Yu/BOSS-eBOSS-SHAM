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

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
date     = sys.argv[3]
mode     = sys.argv[4]
npoints  = 200#int(sys.argv[3])
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
fileroot = 'MCMCout/'+date+'/HAM_'+gal+'_'+GC+'_'+mode+'/multinest_'
if os.path.exists('MCMCout/'+date+'/HAM_'+gal+'_'+GC+'_'+mode)==False:
    os.makedirs('MCMCout/'+date+'/HAM_'+gal+'_'+GC+'_'+mode)

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
s = (bins[:-1]+bins[1:])/2

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 

# read the posterior file
parameters = ["sigma","vcut"]
npar = len(parameters)
sample = loadMCSamples(fileroot)
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
    
# plot the best-fit
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(mean[0])),repeat(np.float32(mean[1]))))

if multipole=='mono':    
    fig,ax =plt.subplots(figsize=(8,6))
    ax.errorbar(s,s**2*obscf['col3'],s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
    ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[0],c='m',alpha=0.5)
    label = ['best fit','obs']
    plt.legend(label,loc=0)
    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
    plt.xlabel('s (Mpc $h^{-1}$)')
    plt.ylabel('s^2 * $\\xi_0$')
    plt.savefig('HAM-MCMC_cf_mono_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig =plt.figure(figsize=(16,6))
    for col,covbin,k in zip(['col3','col4'],[int(0),int(200)],range(2)):
        ax = plt.subplot2grid((1,2),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('HAM-MCMC_cf_quad_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole == 'hexa':
    fig =plt.figure(figsize=(24,6))
    for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
        ax = plt.subplot2grid((1,3),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('HAM-MCMC_cf_hexa_bestfit_'+gal+'_'+GC+'.png',bbox_tight=True)
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
	plt.title('{} {} distribution: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
	plt.xlabel(var+' (km/s)')
	ax.set_xlim(1000,10)

	ax = plt.subplot2grid((1,2),(0,1))
	ax.plot(bins[:-1],n2,alpha=0.5,lw=0.5)
	ax.plot(bins[:-1],n,alpha=0.5,lw=0.5)
	plt.yscale('log')
	plt.ylabel('galaxy numbers')
	plt.title('{} {} distribution: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,mean[0],mean[1]))
	plt.xlabel(var+' (km/s)')
	ax.set_xlim(1000,10)


plt.savefig('HAM-MCMC_distr_'gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
