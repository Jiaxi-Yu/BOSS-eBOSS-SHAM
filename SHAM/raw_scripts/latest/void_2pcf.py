import matplotlib 
matplotlib.use('agg')
import sys
import numpy as np
import pandas as pd
sys.path.append('/global/cscratch1/sd/jiaxi/SHAM/codes/pydive/pydive/')
from dive import galaxies_to_voids
import time
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool 
from itertools import repeat
import glob
import h5py

# variables
gal      = 'LRG'
GC       = 'SGC'
date     = '1118'
cut      = 'indexcut'
nseed    = 20
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
home      = '/global/cscratch1/sd/jiaxi/SHAM/'
modes    = ['best_check','close_chi2']
mode     = modes[int(sys.argv[1])]
Rv = [[0,15],[15,30],[30,1000]]
Rvmin    = Rv[int(sys.argv[2])][0]
Rvmax    = Rv[int(sys.argv[2])][1]

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    ver='v7_2'
    halofile = home+'catalog/UNIT_hlist_0.58760.hdf5' 
    #halofile = home+'catalog/UNIT4LRG-cut.hdf5'

if gal == 'ELG':
    SHAMnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    ver='v7'
    halofile = home+'catalog/UNIT_hlist_0.53780.hdf5'
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
print('selecting only the necessary variables...')
f=h5py.File(halofile,"r")
sel = f["halo"]['Vpeak'][:]>200
if len(f["halo"]['Vpeak'][:][sel])%2 ==1:
    datac = np.zeros((len(f["halo"]['Vpeak'][:][sel])-1,5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = (f["halo"][key][:][sel])[:-1]
else:
    datac = np.zeros((len(f["halo"]['Vpeak'][:][sel]),5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = f["halo"][key][:][sel]
f.close()        
half = int(len(datac)/2)


# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    x00,x20= sham_cal(uni,sigM,sigV,Mtrun)
    x01,x21= sham_cal(uni1,sigM,sigV,Mtrun)
    return [append(x00,x01),append(x20,x21)]

def sham_cal(uniform,sigma_high,sigma,v_high):
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
    LRGscat[:,-1]  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    LRGscat[:,-1] %=boxsize
    
    # calculate the 2pcf of the SHAM galaxies
    # count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],LRGscat[:,-1],periodic=True, verbose=True,boxsize=boxsize)
    # calculate the 2pcf and the multipoles
    mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    # void
    void = galaxies_to_voids(LRGscat[:,2:],r_min=Rvmin, r_max=Rvmax)
    DD_countsv = DDsmu(autocorr, nthread,bins,mu_max, nmu,void[:,0],void[:,1],void[:,2],periodic=True, verbose=True,boxsize=boxsize)
    # calculate the 2pcf and the multipoles
    monov = (DD_countsv['npairs'].reshape(nbins,nmu)/(len(void)**2)/rr-1)
    quadv = monov * 2.5 * (3 * mu**2 - 1)
    hexav = monov * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    # use sum to integrate over mu
    return [np.sum(monov,axis=-1)/nmu,np.sum(quadv,axis=-1)/nmu]

if mode == 'best_check':
    # the best-fit parameters
    par  = ["sigma","Vsmear","Vceil"]
    npar = len(par)
    fileroot = '{}MCMCout/indexcut_{}/LRG_SGC_hexa_prior3_5-35/multinest_'.format(home,date)
    a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
    # calculate the best 2pcf
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2])))))
    
        # plot the 2PCF multipoles   
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(['col4','col5'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
            true_mean = ((np.mean(xi1_ELG,axis=0)[k])[:nbins]+(np.mean(xi1_ELG,axis=0)[k])[nbins:])/2
            values=[np.zeros(nbins),true_mean]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                ax[j,k].plot(s,s**2*(true_mean-values[j]),c='c',alpha=0.6)
                ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none")
                plt.xlabel('s (Mpc $h^{-1}$)')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                    label = ['SHAM','PIP obs 1$\sigma$']
                    plt.legend(label,loc=0)
                    plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
        plt.savefig('cf_void_{}_bestfit_{}_{}_Rv{}-{}Mpch-1_{}.png'.format(multipole,gal,GC,Rvmin,Rvmax,mode),bbox_tight=True)
        plt.close()

elif mode == 'close_chi2':
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(0.14864245)),repeat(131.20083923),repeat(np.float32(5.24729866))))) 
    with Pool(processes = nseed) as p:
        xi0_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(0.59092776)),repeat(109.82603879),repeat(np.float32(4.87587909))))) 
        
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(['col4','col5'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
            tmp = [xi1_ELG[a][k] for a in range(nseed)]
            true_array = np.hstack((((np.array(tmp)).T)[:nbins],((np.array(tmp)).T)[nbins:]))
            true_mean = np.mean(true_array,axis=1)
            true_std  = np.std(true_array,axis=1)
            tmp0 = [xi0_ELG[a][k] for a in range(nseed)]
            true_array0 = np.hstack((((np.array(tmp0)).T)[:nbins],((np.array(tmp0)).T)[nbins:]))
            true_mean0 = np.mean(true_array0,axis=1)
            values=[np.zeros(nbins),true_mean]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                ax[j,k].plot(s,s**2*(true_mean0-values[j]),c='m',alpha=0.6)
                ax[j,k].errorbar(s,s**2*(true_mean-values[j]),s**2*true_std,color='c')
                plt.xlabel('s (Mpc $h^{-1}$)')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                    label = ['$\chi^2=75.33$','$\chi^2=75.39$']
                    plt.legend(label,loc=0)
                    plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
        plt.savefig('cf_void_{}_{}_{}_{}_Rv{}-{}Mpch-1.png'.format(multipole,mode,gal,GC,Rvmin,Rvmax),bbox_tight=True)
        plt.close()