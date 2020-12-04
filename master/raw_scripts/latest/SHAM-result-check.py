import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.wp import wp
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import pymultinest
import corner
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
rmin     = 2
rmax     = 20
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home     = '/global/cscratch1/sd/jiaxi/SHAM/'
modes    = ['best_check','close_chi2']
funcs    = ['wp','mps']
mode     = modes[int(sys.argv[1])]
func     = funcs[int(sys.argv[2])]

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    ver='v7_2'
    halofile = home+'catalog/UNIT_hlist_0.58760.hdf5' 

if gal == 'ELG':
    SHAMnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    ver='v7'
    halofile = home+'catalog/UNIT_hlist_0.53780.hdf5'
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate separation bins
if func == 'mps':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    
    # generate mu bins   
    s = (bins[:-1]+bins[1:])/2
    mubins = np.linspace(0,mu_max,nmu+1)
    mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

    # Analytical RR calculation
    RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
    rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
    print('the analytical random pair counts are ready.')
    
    # covariance matrices and observations
    obs2pcf = '{}catalog/nersc_mps_{}_{}/mps_{}_{}_{}.dat'.format(home,gal,ver,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/mps_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,rscale,gal,multipole)
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) # cov([mono,quadru])
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    errbar = np.std(hdu[1].data[GC+'mocks'],axis=1)
    hdu.close()
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:]))
    covcut  = cov(mocks).astype('float32')
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax] 
    OBS   = append(obscf['col4'],obscf['col5']).astype('float32')
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
elif func == 'wp':
    # zbins with log binned mps and wp
    covfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(home,func,gal,ver,func,gal) 
    obs2pcf  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver)
    # bin
    binfile = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    # observation
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf = obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    OBS   = obscf['col4']
    bins  = np.unique(append(obscf['col1'],obscf['col2']))
    nbins = len(bins)-1
    s = obscf['col3']
    OBS   = np.array(obscf['col4']).astype('float32')
    # Read the covariance matrices
    hdu = fits.open(covfits) 
    mocks = hdu[1].data[GC+'mocks'][binmin:binmax,:]
    Nmock = mocks.shape[1] 
    errbar = np.std(mocks,axis=1)
    hdu.close()
    covcut  = cov(mocks).astype('float32')
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)

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
    if func == 'mps':
        x00,x20= sham_cal(uni,sigM,sigV,Mtrun)
        x01,x21= sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = [(x00+x01)/2,(x20+x21)/2]
    else:        
        x00    = sham_cal(uni,sigM,sigV,Mtrun)
        x01    = sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = (x00+x01)/2
    return tpcf

def sham_cal(uniform,sigma_high,sigma,v_high):
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    if cut=='aftercut':
        LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],SHAMnum)[:(SHAMnum)]]
    elif cut =='precut':
        LRGscat = (datac[datac[:,-1]<v_high])[argpartition(-datav[datac[:,-1]<v_high],SHAMnum)[:(SHAMnum)]]
    elif cut=='indexcut':
        LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
        datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
        LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
        datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    else:
        print('wrong input!')
        
    n,BINS = np.histogram(LRGscat[:,1],range =(0,1500),bins=100)
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
    LRGscat[:,-1]  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    LRGscat[:,-1] %=boxsize
    
    # calculate the 2pcf of the SHAM galaxies
    # count the galaxy pairs and normalise them
    if func == 'mps':
        DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],LRGscat[:,-1],periodic=True, verbose=True,boxsize=boxsize)
        # calculate the 2pcf and the multipoles
        mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
        quad = mono * 2.5 * (3 * mu**2 - 1)
        # use sum to integrate over mu
        SHAM_array = [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu]
    else:
        wp_dat = wp(boxsize,80,nthread,bins,LRGscat[:,2],LRGscat[:,3],LRGscat[:,-1])#,periodic=True, verbose=True)
        SHAM_array = wp_dat['wp']
    return SHAM_array

# calculate the SHAM 2PCF
if mode == 'best_check':
    # the best-fit parameters
    par  = ["sigma","Vsmear","Vceil"]
    npar = len(par)
    fileroot = '{}MCMCout/indexcut_{}/LRG_SGC_hexa_prior3_5-35/multinest_'.format(home,date)
    a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
    # calculate the best 2pcf
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2])))))
        
    # plot the observation and SHAM models
    if func == 'wp':
        WP = np.loadtxt('{}/catalog/2PCF_obs/wp/{}_{}.dat'.format(home,gal,GC))
        #OBS2  = 2*np.sum(pairs.reshape(30,80),axis=-1)
        OBS2  = WP[binmin:binmax]
        # plot the 2PCF multipoles   
        fig = plt.figure(figsize=(5,6))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,1), dtype=type(plt.axes))
        k=0
        values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,(np.mean(xi1_ELG,axis=0)-values[j]),c='c',alpha=0.6)
            ax[j,k].errorbar(s,(OBS-values[j]),errbar,color='k', marker='o',ecolor='k',ls="none",markersize = 4)
            ax[j,k].errorbar(s,(OBS2[:,1]-values[j]),errbar,color='r', marker='o',ecolor='r',ls="none",markersize = 4)
            plt.xlabel('s (Mpc $h^{-1}$)')
            plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                label = ['SHAM','PIP obs 1$\sigma$','CP obs 1$\sigma$']
                plt.legend(label,loc=0)
                plt.title('projected 2-point correlation function: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
        plt.savefig('{}_bestfit_{}_{}_{}-{}Mpch-1_{}.png'.format(func,gal,GC,rmin,rmax,mode),bbox_tight=True)
        plt.close()
    else:
        OBS2 = np.loadtxt('{}/catalog/2PCF_obs/mps/{}_{}.dat'.format(home,gal,GC))[binmin:binmax]
        # plot the 2PCF multipoles   
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(['col4','col5'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)[k]]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
                ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none")
                ax[j,k].errorbar(s,s**2*(OBS2[:,k+1]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='r', marker='o',ecolor='r',ls="none")
                plt.xlabel('s (Mpc $h^{-1}$)')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                    label = ['SHAM','PIP obs 1$\sigma$','CP obs 1$\sigma$']
                    plt.legend(label,loc=0)
                    plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
        plt.savefig('cf_{}_bestfit_{}_{}_{}-{}Mpch-1_{}.png'.format(multipole,gal,GC,rmin,rmax,mode),bbox_tight=True)
        plt.close()
elif mode == 'close_chi2':
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(0.14864245)),repeat(131.20083923),repeat(np.float32(5.24729866))))) 

    with Pool(processes = nseed) as p:
        xi0_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(0.59092776)),repeat(109.82603879),repeat(np.float32(4.87587909))))) 
    
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(['col4','col5'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)[k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
            ax[j,k].plot(s,s**2*(np.mean(xi0_ELG,axis=0)[k]-values[j]),c='m',alpha=0.6)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                label = ['$\chi^2=75.39$','$\chi^2=75.33$']
                plt.legend(label,loc=0)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
    plt.savefig('cf_{}_bestfit_{}_{}_{}-{}Mpch-1_{}.png'.format(multipole,gal,GC,rmin,rmax,mode),bbox_tight=True)
    plt.close()

"""
np.savetxt(bestfit,np.vstack((np.mean(xi1_ELG,axis=0)[0],np.mean(xi1_ELG,axis=0)[1],np.mean(xi1_ELG,axis=0)[2])).T)
print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[3]))
a,b = np.histogram(np.ones(5000),range =(0,1500),bins=100)
fig,ax = plt.subplots()
plt.plot((b[1:]+b[:-1])/2,np.mean(xi1_ELG,axis=0)[4], label='$\chi^2=75.39$')
plt.plot((b[1:]+b[:-1])/2,np.mean(xi0_ELG,axis=0)[4], label='$\chi^2=75.33$')
plt.legend(loc=0)
plt.ylabel('number of halos')
plt.xlabel('Vpeak(km/s)')
plt.savefig('hist.png')
plt.close()
"""