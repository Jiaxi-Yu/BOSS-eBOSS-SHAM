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
zmin     = sys.argv[4]
zmax     = sys.argv[5]
pre      = sys.argv[6]
function = 'mps' # 'wp'
date     = '0218' 
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
nthread  = 1
autocorr = 1
mu_max   = 1
autocorr = 1
home     = '/home/astro/jiayu/Desktop/SHAM/'
fileroot = '{}MCMCout/zbins_{}/{}{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,pre,function,rscale,gal,GC,zmin,zmax)
bestfit   = '{}bestfit_{}_{}.dat'.format(fileroot[:-10],function,date)
cols = ['col4','col5']

# wp binning:
smin=5; smax=30
binfilewp = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
selwp = (binfilewp['col3']<smax)&(binfilewp['col3']>=smin)
binswp  = np.unique(np.append(binfilewp['col1'][selwp],binfilewp['col2'][selwp]))
swp = binfilewp['col3'][selwp]
nbinswp = len(swp)
nmu      = 80

# read the posterior file
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

# SHAMnum, z, a_t, and version
if (rscale=='linear')&(function=='mps'):
    if gal == 'LRG':
        SHAMnum   = int(6.26e4)
        z = 0.7018
        a_t = '0.58760'
        ver = 'v7_2'
    elif gal=='ELG':
        SHAMnum   = int(2.93e5)
        z = 0.8594
        a_t = '0.53780'
        ver = 'v7'
        cols = ['col3','col4']
    elif gal=='CMASSLOWZTOT':
        SHAMnum = 208000
        z = 0.5609
        a_t = '0.64210'
    elif gal=='CMASS':
        if (zmin=='0.43')&(zmax=='0.51'): 
            SHAMnum = 342000
            z = 0.4686
            a_t = '0.68620'
        elif zmin=='0.51':
            SHAMnum = 363000
            z = 0.5417 
            a_t = '0.64210'
        elif zmin=='0.57':
            SHAMnum = 160000
            z = 0.6399
            a_t =  '0.61420'
        elif (zmin=='0.43')&(zmax=='0.7'):            
            SHAMnum = 264000
            z = 0.5897
            a_t = '0.62800'
    elif gal=='LOWZ':
        if (zmin=='0.2')&(zmax=='0.33'):            
            SHAMnum = 337000
            z = 0.2754
            a_t = '0.78370' 
        elif zmin=='0.33':
            SHAMnum = 258000
            z = 0.3865
            a_t = '0.71730'
        elif (zmin=='0.2')&(zmax=='0.43'): 
            SHAMnum = 295000
            z = 0.3441
            a_t = '0.74980'
    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    if (gal == 'LRG')|(gal=='ELG'):
        obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
        covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
        obstool = 'PIP'
    else:
        obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.mps'.format(home,gal,zmin,zmax)
        covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
        obstool = ''
    
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    if (gal == 'LRG')|(gal=='ELG'):
        Nstot=200
    else:
        Nstot=100
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
    covcut  = cov(mocks).astype('float32')
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
    if gal == 'ELG':
        OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
    else:
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    print('the covariance matrix and the observation 2pcf vector are ready.')
    
elif (rscale=='log'):
    # read s bins
    binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header');ver1='v7_2'
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

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# SHAM halo catalogue
if os.path.exists('{}best-fit-wp_{}_{}-python_pi35.dat'.format(fileroot[:-10],gal,GC)):
    wp = [np.loadtxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,i)) for i in pimaxs]
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
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2]))))) 

    # wp pimaxs calculations for SHAM, observation and mocks
    wp = [0,1,2]
    for h,pimax in enumerate(pimaxs):
        # Corrfunc
        tmp = [xi1_ELG[a][h] for a in range(nseed)]
        true_array = np.hstack((((np.array(tmp)).T)[:nbinswp],((np.array(tmp)).T)[nbinswp:]))
        wp[h]= (np.array([swp,np.mean(true_array,axis=1),np.std(true_array,axis=1)]).reshape(3,nbinswp)).T
        np.savetxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,pimax),wp[h],header='s wp wperr')
        
        # observations
        if (gal == 'LRG')|(gal=='ELG'):
            if zmin =='0.65':
                pairfile = '{}catalog/nersc_zbins_wp_mps_{}/pairs_rp-pi_log_eBOSS_{}_{}_{}_pip_zs_{}-{}0.dat'.format(home,gal,gal,GC,verwp,zmin,zmax)
            else:
                pairfile = '{}catalog/nersc_zbins_wp_mps_{}/pairs_rp-pi_log_eBOSS_{}_{}_{}_pip_zs_{}0-{}0.dat'.format(home,gal,gal,GC,verwp,zmin,zmax)

            minbin,maxbin,dds,drs,rrs = np.loadtxt(pairfile,unpack=True) 
            sel = maxbin<pimax
            OBSwp1 = np.sum(((((dds-2*drs+rrs)/rrs)[sel]).reshape(33,pimax)),axis=1)*2 
            mins,maxs,mids = np.loadtxt('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),unpack=True,usecols=(0,1,2)) 
            arr = np.array([mins,maxs,mids,OBSwp1]).reshape(4,33).T 
            np.savetxt('{}catalog/nersc_zbins_wp_mps_{}/wp_rp_pip_eBOSS_{}_{}_{}_{}-{}_pi{}.dat'.format(home,gal,gal,GC,ver,zmin,zmax,pimax),arr)
        else:
            pairfile = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}.xi'.format(home,gal,zmin,zmax)
            minbin,maxbin,pibin1,pibin2,xiwp = np.loadtxt(pairfile,unpack=True)
            monocut = (xiwp.reshape(nmu,nbinswp))[:pimax,:]
            mono = np.sum(monocut,axis=0)*2
            minbin,maxbin = np.unique(minbin),np.unique(maxbin)
            midbin = 10**((np.log10(minbin)+np.log10(maxbin))/2)
            arr = np.array([midbin,minbin,maxbin,mono]).reshape(4,nbinswp).T 
            np.savetxt('{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}_pi{}.wp'.format(home,gal,zmin,zmax,pimax),arr)

# wp pimax plot
for h,pimax in enumerate(pimaxs):
    # SHAM wp
    pythonsel = (wp[h][:,0]>smin)&(wp[h][:,0]<smax)
    #import pdb;pdb.set_trace()
    wp[h] = wp[h][pythonsel,:]

    # observations
    if (gal == 'LRG')|(gal=='ELG'):
        if rscale == 'linear':
            #covfitswp  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(home,'wp',gal,ver,'wp',gal) 
            obs2pcfwp  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}_pi{}.dat'.format(home,gal,ver,gal,GC,ver,pimax)
        elif rscale == 'log':
            #covfitswp = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,'wp',rscale,gal,zmin,zmax,multipole) 
            obs2pcfwp  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}_pi{}.dat'.format(home,gal,'wp',rscale,gal,GC,ver1,zmin,zmax,pimax)
        obstool = 'PIP'
        colwp   = 'col3'
    else:
        obs2pcfwp = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}_pi{}.wp'.format(home,gal,zmin,zmax,pimax)
        covfitswp = '{}catalog/BOSS_zbins_wp/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,gal,zmin,zmax,pimax)
        obstool = ''
        colwp   = 'col1'
    # observation rp selection
    obscfwp = Table.read(obs2pcfwp,format='ascii.no_header')
    selwp = (obscfwp[colwp]<smax)&(obscfwp[colwp]>=smin)
    OBSwp   = obscfwp['col4'][selwp]

    # Read the covariance matrices
    hdu = fits.open(covfitswp) 
    mockswp = hdu[1].data[GC+'mocks'][pythonsel,:]
    Nmockwp = mockswp.shape[1] 
    errbarwp = np.std(mockswp,axis=1)
    hdu.close()  

    # plot
    fig = plt.figure(figsize=(6,7))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    #import pdb;pdb.set_trace()
    for k in range(1):
        values=[np.zeros_like(OBSwp),OBSwp]
        err   = [np.ones_like(OBSwp),errbarwp]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k]);#import pdb;pdb.set_trace()
            ax[j,k].errorbar(swp,(OBSwp-values[j])/err[j],errbarwp/err[j],color='k', marker='D',ecolor='k',ls="none",label='obs+mocks $\pi${}'.format(pimax))
            ax[j,k].plot(swp,(wp[h][:,1]-values[j])/err[j],color='b',label='SHAM $\pi${}'.format(pimax))
            #ax[j,k].errorbar(swp,(obscfwp-values[j])/err[j],errbarwp/err[j],color='k', marker='o',ecolor='k',ls="none",label='PIP obs 1$\sigma$')
            plt.xlabel('rp (Mpc $h^{-1}$)')
            plt.xscale('log')
            if (j==0):        
                plt.yscale('log')
                ax[j,k].set_ylabel('wp')
                plt.legend(loc=0)
                plt.title('projected 2pcf: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta$ wp/err')
                plt.ylim(-3,3)
    plt.savefig('{}wp_bestfit_{}_{}_z{}z{}_{}-{}Mpch-1_pi{}.png'.format(fileroot[:-10],gal,GC,zmin,zmax,smin,smax,pimax),bbox_tight=True)
    plt.close()