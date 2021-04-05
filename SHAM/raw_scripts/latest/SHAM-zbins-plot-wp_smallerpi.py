#!/usr/bin/env python3
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
import warnings
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
gal      = sys.argv[1]
GC       = sys.argv[2]
rscale   = sys.argv[3] #'linear' # 'log'
function = 'mps' # 'wp'
date    = sys.argv[4]
zmin     = sys.argv[5]
zmax     = sys.argv[6]
finish   = int(sys.argv[7])
nseed    = 15
pimaxs = [25,30,35]
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
if rscale =='linear':
    rmax = 25
else:
    rmax = 30
nthread  = 32
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=35
home     = '/home/astro/jiayu/Desktop/SHAM/'
fileroot = '{}MCMCout/zbins_{}/{}{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,sys.argv[8],function,rscale,gal,GC,zmin,zmax)
bestfit   = '{}bestfit_{}_{}.dat'.format(fileroot[:-10],function,date)
cols = ['col4','col5']

# read the posterior file
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

if (rscale=='linear')&(function=='mps'):
    if gal == 'LRG':
        SHAMnum   = int(6.26e4)
        z = 0.7018
        a_t = '0.58760'
        ver = 'v7_2'
    else:
        SHAMnum   = int(2.93e5)
        z = 0.8594
        a_t = '0.53780'
        ver = 'v7'
        cols = ['col3','col4']
       
    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:]))
    covcut  = cov(mocks).astype('float32')
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
    if gal == 'LRG':
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')
    else:
        OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
        
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    print('the covariance matrix and the observation 2pcf vector are ready.')
    
elif (rscale=='log'):
    # read s bins
    binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
    ver1='v7_2';verwp = 'v7'
    sel = (binfile['col3']<rmax)&(binfile['col3']>=rmin)
    bins  = np.unique(np.append(binfile['col1'][sel],binfile['col2'][sel]))
    s = binfile['col3'][sel]
    nbins = len(bins)-1
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1

    if gal == 'LRG':
        ver = 'v7_2'
        extra = np.ones_like(s)
        #extra = binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]**2
    else:
        ver = 'v7'
        extra = np.ones_like(s)
    # filenames
    covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
    obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1]
    hdu.close()
    # observations
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance and errobar for chi2
    Ns = int(mocks.shape[0]/2)
    mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
    covcut  = cov(mocks).astype('float32')
    OBS   = append(obscf['col4']/extra,obscf['col5']/extra).astype('float32')# LRG columns are s**2*xi
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)

    # zbins, z_eff ans ngal
    if (zmin=='0.6')&(zmax=='0.8'):
        if gal=='ELG':
            SHAMnum = int(3.26e5)
            z = 0.7136
        else:
            SHAMnum = int(8.86e4)
            z = 0.7051
        a_t = '0.58760'
    elif (zmin=='0.6')&(zmax=='0.7'):            
        SHAMnum = int(9.39e4)
        z = 0.6518
        a_t = '0.60080'
    elif zmin=='0.65':
        SHAMnum = int(8.80e4)
        z = 0.7273
        a_t = '0.57470'
    elif zmin=='0.9':
        SHAMnum = int(1.54e5)
        z = 0.9938
        a_t = '0.50320'
    elif zmin=='0.7':
        if gal=='ELG':
            SHAMnum = int(4.38e5)
            z = 0.8045# To be calculated
        else:
            SHAMnum = int(6.47e4)
            z=0.7968
        a_t = '0.54980'
    else:
        if gal=='ELG':
            SHAMnum = int(3.34e5)
            z = 0.9045 # To be calculated
        else:
            SHAMnum = int(3.01e4)
            z= 0.8777
        a_t = '0.52600'
else:
    print('wrong 2pcf function input')


# wp plot
if rscale == 'linear':
    covfitswp  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(home,'wp',gal,ver,'wp',gal) 
    obs2pcfwp  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver)

elif rscale == 'log':
    covfitswp = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,'wp',rscale,gal,zmin,zmax,multipole) 
    obs2pcfwp  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,'wp',rscale,gal,GC,ver1,zmin,zmax)


binfilewp = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
selwp = (binfilewp['col3']<smax)&(binfilewp['col3']>=smin)
binswp  = np.unique(np.append(binfilewp['col1'][selwp],binfilewp['col2'][selwp]))
swp = binfilewp['col3'][selwp]
binminwp = np.where(binfilewp['col3']>=smin)[0][0]
binmaxwp = np.where(binfilewp['col3']<smax)[0][-1]+1
nbinswp = len(binswp)-1
# observation
obscfwp = Table.read(obs2pcfwp,format='ascii.no_header')
selwp = (obscfwp['col3']<smax)&(obscfwp['col3']>=smin)
OBSwp   = obscfwp['col4'][selwp]


# Read the covariance matrices
"""
hdu = fits.open(covfitswp) 
mockswp = hdu[1].data[GC+'mocks'][binminwp:binmaxwp,:]
Nmockwp = mockswp.shape[1] 
errbarwp = np.std(mockswp,axis=1)
hdu.close()
"""

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# SHAM halo catalogue
if os.path.exists('{}best-fit-wp_{}_{}-python_pi35.dat'.format(fileroot[:-10],gal,GC)):
    wp = [np.loadtxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,i)) for i in pimaxs]
    if rscale == 'linear':
        wp80 = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
    elif rscale == 'log':
        wp80 = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))[10:]

else:
    print('reading the UNIT simulation snapshot with a(t)={}'.format(a_t))  
    halofile = home+'catalog/UNIT_hlist_'+a_t+'.hdf5'        
    read = time.time()
    f=h5py.File(halofile,"r")
    sel = f["halo"]['Vpeak'][:]>0
    if len(f["halo"]['Vpeak'][:][sel])%2 ==1:
        datac = np.zeros((len(f["halo"]['Vpeak'][:][sel])-1,5))
        for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = (f["halo"][key][:][sel])[:-1]
    else:
        datac = np.zeros((len(f["halo"]['Vpeak'][:][sel]),5))
        for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = f["halo"][key][:][sel]
    f.close()        
    half = int32(len(datac)/2)
    print(len(datac))
    print('read the halo catalogue costs {:.6}s'.format(time.time()-read))

    # generate uniform random numbers
    print('generating uniform random number arrays...')
    uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
    uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

    # SHAM application
    def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
        wp00,wp01,wp02= sham_cal(uni,sigM,sigV,Mtrun)
        wp10,wp11,wp12= sham_cal(uni1,sigM,sigV,Mtrun)
        return [append(wp00,wp10),append(wp01,wp11),append(wp02,wp12)]

    def sham_cal(uniform,sigma_high,sigma,v_high):
        # scatter Vpeak
        scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
        scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
        datav = datac[:,1]*scatter
        # select halos
        percentcut = int(len(datac)*v_high/100)
        LRGscat = datac[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
        datav = datav[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
        LRGscat = LRGscat[argpartition(-datav,percentcut)[percentcut:]]
        datav = datav[argpartition(-datav,percentcut)[percentcut:]]
        # binnning Vpeak of the selected halos
        
        # transfer to the redshift space
        scathalf = int(len(LRGscat)/2)
        z_redshift  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
        z_redshift %=boxsize
        
        # Corrfunc 2pcf and wp
        #wp_dat80 = wp(boxsize,80,nthread,binswp,LRGscat[:,2],LRGscat[:,3],z_redshift)
        wp_dat0 = wp(boxsize,pimaxs[0],nthread,binswp,LRGscat[:,2],LRGscat[:,3],z_redshift)
        wp_dat1 = wp(boxsize,pimaxs[1],nthread,binswp,LRGscat[:,2],LRGscat[:,3],z_redshift)
        wp_dat2 = wp(boxsize,pimaxs[2],nthread,binswp,LRGscat[:,2],LRGscat[:,3],z_redshift)


        # calculate the 2pcf and the multipoles
        return [wp_dat0['wp'],wp_dat1['wp'],wp_dat2['wp']]

    # calculate the SHAM 2PCF
    if finish: 
        with Pool(processes = nseed) as p:
            xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2]))))) 
        # wp80
        #tmp = [xi1_ELG[a][0] for a in range(nseed)]
        #true_array = np.hstack((((np.array(tmp)).T)[:nbinswp],((np.array(tmp)).T)[nbinswp:]))
        #wp80= (np.array([swp,np.mean(true_array,axis=1),np.std(true_array,axis=1)]).reshape(3,nbinswp)).T

        # wp pimaxs
        wp = [0,1,2]
        for h,pimax in enumerate(pimaxs):
            # Corrfunc
            tmp = [xi1_ELG[a][h] for a in range(nseed)]
            true_array = np.hstack((((np.array(tmp)).T)[:nbinswp],((np.array(tmp)).T)[nbinswp:]))
            wp[h]= (np.array([swp,np.mean(true_array,axis=1),np.std(true_array,axis=1)]).reshape(3,nbinswp)).T
            np.savetxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,pimax),wp[h],header='s wp wperr')
            if zmin =='0.65':
                pairfile = '{}catalog/nersc_zbins_wp_mps_{}/pairs_rp-pi_log_eBOSS_{}_{}_{}_pip_zs_{}-{}0.dat'.format(home,gal,gal,GC,verwp,zmin,zmax)
            else:
                pairfile = '{}catalog/nersc_zbins_wp_mps_{}/pairs_rp-pi_log_eBOSS_{}_{}_{}_pip_zs_{}0-{}0.dat'.format(home,gal,gal,GC,verwp,zmin,zmax)

            minbin,maxbin,dds,drs,rrs = np.loadtxt(pairfile,unpack=True) 
            sel = maxbin<pimax
            OBSwp1 = np.sum(((((dds-2*drs+rrs)/rrs)[sel]).reshape(33,pimax)),axis=1)*2 
            mins,maxs,mids,wp80s = np.loadtxt('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),unpack=True) 
            arr = np.array([mins,maxs,mids,OBSwp1]).reshape(4,33).T 
            np.savetxt('{}catalog/nersc_zbins_wp_mps_{}/wp_rp_pip_eBOSS_{}_{}_{}_{}-{}_pi{}.dat'.format(home,gal,gal,GC,ver,zmin,zmax,pimax),arr)
            OBSwp1 = OBSwp1[selwp]
        
# wp pimax 
for h,pimax in enumerate(pimaxs):
    # obs
    if rscale == 'linear':
        obs2pcfwp1  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}_pi{}.dat'.format(home,gal,ver,gal,GC,ver,pimax)
        OBSwp1   = Table.read(obs2pcfwp1,format='ascii.no_header')['col4'][selwp]
    elif rscale =='log':
        obs2pcfwp1 = '{}catalog/nersc_zbins_wp_mps_{}/wp_rp_pip_eBOSS_{}_{}_{}_{}-{}_pi{}.dat'.format(home,gal,gal,GC,ver,zmin,zmax,pimax)
        OBSwp1   = Table.read(obs2pcfwp1,format='ascii.no_header')['col4'][selwp]

    # plot
    fig = plt.figure(figsize=(6,7))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    #import pdb;pdb.set_trace()
    for k in range(1):
        values=[np.zeros_like(OBSwp),OBSwp1]
        err   = [np.ones_like(OBSwp),wp[h][:,2]]

        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k]);#import pdb;pdb.set_trace()
            ax[j,k].errorbar(swp,(wp[h][:,1]-values[j])/err[j],wp[h][:,2]/err[j],color='k', marker='D',ecolor='k',ls="none",label='SHAM_pi{}'.format(pimax))
            ax[j,k].plot(swp,(OBSwp1-values[j])/err[j],color='b',label='PIP_pi{}'.format(pimax))
            #ax[j,k].errorbar(swp,(obscfwp-values[j])/err[j],errbarwp/err[j],color='k', marker='o',ecolor='k',ls="none",label='PIP obs 1$\sigma$')
            plt.xlabel('rp (Mpc $h^{-1}$)')
            plt.xscale('log')
            if (j==0):        
                plt.yscale('log')
                ax[j,k].set_ylabel('wp')
                plt.legend(loc=0)
                plt.title('projected 2pcf: {} in {}, errorbar from SHAM'.format(gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta$ wp/err')
                plt.ylim(-3,3)

    plt.savefig('{}wp_bestfit_{}_{}_z{}z{}_{}-{}Mpch-1_pi{}.png'.format(fileroot[:-10],gal,GC,zmin,zmax,smin,smax,pimax),bbox_tight=True)
    plt.close()

"""
# wp 80
fig = plt.figure(figsize=(6,7))
spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
#import pdb;pdb.set_trace()
for k in range(1):
    values=[np.zeros_like(OBSwp),OBSwp]
    err   = [np.ones_like(OBSwp),wp80[:,2]]

    for j in range(2):
        ax[j,k] = fig.add_subplot(spec[j,k]);#import pdb;pdb.set_trace()
        ax[j,k].errorbar(swp,(wp80[:,1]-values[j])/err[j],wp80[:,2]/err[j],color='k', marker='D',ecolor='k',ls="none",label='SHAM_pi80')
        ax[j,k].plot(swp,(OBSwp-values[j])/err[j],color='b',label='PIP_pi80'.format(pimax))
        #ax[j,k].errorbar(swp,(obscfwp-values[j])/err[j],errbarwp/err[j],color='k', marker='o',ecolor='k',ls="none",label='PIP obs 1$\sigma$')
        plt.xlabel('rp (Mpc $h^{-1}$)')
        plt.xscale('log')
        if (j==0):        
            plt.yscale('log')
            ax[j,k].set_ylabel('wp')
            plt.legend(loc=0)
            plt.title('projected 2pcf: {} in {}, errorbar from SHAM'.format(gal,GC))
        if (j==1):
            ax[j,k].set_ylabel('$\Delta$ wp/err')
            plt.ylim(-3,3)

plt.savefig('{}wp_bestfit_{}_{}_z{}z{}_{}-{}Mpch-1.png'.format(fileroot[:-10],gal,GC,zmin,zmax,smin,smax),bbox_tight=True)
plt.close()
"""