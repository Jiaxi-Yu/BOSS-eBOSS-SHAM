#!/usr/bin/env python3
import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
import os
import warnings
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import pymultinest
import corner
import h5py

# variables
gal      = sys.argv[1]
GC       = 'NGC+SGC'
if gal == 'LRG':
    rscale   = 'log'
else:
    rscale = 'linear'
function = 'mps' 
nseed    = 30
date     = '0218'
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
smin=5; smax=30
home     = '/home/astro/jiayu/Desktop/SHAM/'
if gal == 'LRG':
    zmins = [0.6,0.6,0.65,0.7,0.8]
    zmaxs = [0.7,0.8,0.8, 0.9,1.0]
    ver = 'v7_2'
elif gal == 'ELG':
    zmins = [0.6,0.7,0.8,0.9]
    zmaxs = [0.8,0.9,1.0,1.1]
    ver = 'v7'
elif gal == 'CMASS':
    zmins = [0.43,0.51,0.57]
    zmaxs = [0.51,0.57,0.7]
elif gal == 'LOWZ':
    zmins = [0.2, 0.33]
    zmaxs = [0.33,0.43]
zbinnum = len(zmins)
colors = ['m','b','orange','r','c']
cols = ['col4','col5']
ver1 = 'v7_2'

samples = [x for x in range(zbinnum)]
bestfits = [x for x in range(zbinnum)]
Ccodes = [x for x in range(zbinnum)]
obscfs= [x for x in range(zbinnum)]
errbars= [x for x in range(zbinnum)]
OBSwps= [x for x in range(zbinnum)]
wps= [x for x in range(zbinnum)]
errbarwps= [x for x in range(zbinnum)]
pdfs= [x for x in range(zbinnum)]

########################################################
for zbin in range(zbinnum):
    zmin,zmax = zmins[zbin],zmaxs[zbin]
    # getdist results
    fileroot = '{}MCMCout/zbins_{}/{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,function,rscale,gal,GC,zmin,zmax)
    parameters = ["sigma","Vsmear","Vceil"]
    npar = len(parameters)
    a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
    sample = loadMCSamples(fileroot)

    # read s bins
    if (gal == 'LRG')|(gal=='ELG'):
        binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
        sel = (binfile['col3']<rmax)&(binfile['col3']>=rmin)
        bins  = np.unique(np.append(binfile['col1'][sel],binfile['col2'][sel]))
        s = binfile['col3'][sel]
        nbins = len(bins)-1
        binmin = np.where(binfile['col3']>=rmin)[0][0]
        binmax = np.where(binfile['col3']<rmax)[0][-1]+1
        # filenames
        covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
        obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
    else:
        # generate s bins
        bins  = np.arange(rmin,rmax+1,1)
        nbins = len(bins)-1
        binmin = rmin
        binmax = rmax
        s = (bins[:-1]+bins[1:])/2
        # filenames
        obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.mps'.format(home,gal,zmin,zmax)
        covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
        obstool = ''
        
    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1]
    hdu.close()
    if gal == 'LRG':
        Nstot = int(mocks.shape[0]/2)
    else: 
        Nstot = 100

    # observations
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    if rscale == 'linear':
        obscf= obscf[(obscf['col1']<rmax)&(obscf['col1']>=rmin)]
    else:
        obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance and errobar for chi2
    Ns = int(mocks.shape[0]/2)
    mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
    covcut  = cov(mocks).astype('float32')
    OBS   = append(obscf['col4'],obscf['col5']).astype('float32')# LRG columns are s**2*xi
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    errbar = np.std(mocks,axis=1)

    # plot wp with errorbars
    wp = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
    if (gal == 'LRG')|(gal=='ELG'):
        if rscale == 'linear':
            covfitswp  = '{}catalog/nersc_{}_{}_{}/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,'wp',gal,ver,gal,zmin,zmax) 
            obs2pcfwp  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver)
        elif rscale == 'log':
            covfitswp = '{}catalog/nersc_zbins_wp_mps_{}/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,gal,gal,zmin,zmax) 
            obs2pcfwp  = '{}catalog/nersc_zbins_wp_mps_{}/wp_log_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,gal,GC,ver1,zmin,zmax)
        colwp   = 'col3'
    else:
        obs2pcfwp = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}.wp'.format(home,gal,zmin,zmax)
        covfitswp = '{}catalog/BOSS_zbins_wp/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,gal,zmin,zmax)
        colwp   = 'col1'
    pythonsel = (wp[:,0]>smin)&(wp[:,0]<smax)
    wp = wp[tuple(pythonsel),:]
    # observation
    obscfwp = Table.read(obs2pcfwp,format='ascii.no_header')
    selwp = (obscfwp[colwp]<smax)&(obscfwp[colwp]>=smin)
    OBSwp   = obscfwp['col4'][selwp]
    # Read the covariance matrices
    hdu = fits.open(covfitswp) 
    mockswp = hdu[1].data[GC+'mocks']#[binminwp:binmaxwp,:]
    Nmockwp = mockswp.shape[1] 
    errbarwp = np.std(mockswp,axis=1)
    hdu.close()  

    # SHAM halo catalogue
    if rscale=='linear':
        xi = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[binmin:binmax]
    else:
        xi = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[1:]
    bbins,UNITv,SHAMv = np.loadtxt('{}/best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),unpack=True)
    pdf = SHAMv[:-1]/UNITv[:-1]
    
    #print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[2]))
    
    # data summary
    samples[zbin] = sample
    Ccodes[zbin] = xi
    obscfs[zbin] = obscf
    errbars[zbin] = errbar
    OBSwps[zbin] = OBSwp
    wps[zbin]     = wp    
    errbarwps[zbin] = errbarwp
    pdfs[zbin]  = pdf
    bestfits[zbin] = a.get_best_fit()

# plot posteriors
plt.rcParams['text.usetex'] = False
g = plots.getSinglePlotter()
g.settings.figure_legend_frame = False
g.settings.legend_fontsize = 14
g.settings.alpha_filled_add=0.2
g = plots.getSubplotPlotter(subplot_size=3)
g.triangle_plot(samples,parameters,filled=True,\
    legend_labels=['z{}z{}'.format(zmins[x],zmaxs[x]) for x in range(zbinnum)],\
        contour_colors=colors[:zbinnum+1])
for zbin in range(zbinnum):
    for yi in range(npar): 
        for xi in range(yi):
            ax = g.subplots[yi,xi]
            ax.axvline(bestfits[zbin]['parameters'][xi], color=colors[zbin])
            ax.axhline(bestfits[zbin]['parameters'][yi], color=colors[zbin])
            ax.plot(bestfits[zbin]['parameters'][xi],bestfits[zbin]['parameters'][yi], "*",markersize=5,color='k') 
g.export('{}ztot_{}_{}_{}_posterior.png'.format(home,date,gal,GC))
plt.close()

# plot the 2PCF multipoles   
fig = plt.figure(figsize=(14,8))
spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
ax = np.empty((2,2), dtype=type(plt.axes))
for zbin in range(zbinnum):
    for col,covbin,name,k in zip(cols,[int(0),int(Nstot)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscfs[zbin][col]]    
        err   = [np.ones(nbins),s**2*errbars[zbin][k*nbins:(k+1)*nbins]]

        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,s**2*(Ccodes[zbin][:,k+2]-values[j])/err[j],c=colors[zbin],alpha=0.6,label='_hidden')
            if k==0:
                ax[j,k].errorbar(s,s**2*(obscfs[zbin][col]-values[j])/err[j],s**2*errbars[zbin][k*nbins:(k+1)*nbins]/err[j],\
                    color=colors[zbin], marker='o',ecolor=colors[zbin],ls="none",\
                    label='z{}z{}'.format(zmins[zbin],zmaxs[zbin]))
            else:
                ax[j,k].errorbar(s,s**2*(obscfs[zbin][col]-values[j])/err[j],s**2*errbars[zbin][k*nbins:(k+1)*nbins]/err[j],\
                    color=colors[zbin], marker='o',ecolor=colors[zbin],ls="none",\
                    label='z{}z{}, $\chi^2/dof={:.4}$/{}'.format(zmins[zbin],zmaxs[zbin],-2*bestfits[zbin]['log_likelihood'],int(2*len(s)-3)))

            plt.xlabel('s (Mpc $h^{-1}$)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                if k==0:
                    plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('correlation function {}: {} in {}, PIP with errorbar'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2))
                plt.ylim(-3,3)

plt.savefig('{}cf_{}_bestfit_{}_{}_{}-{}Mpch-1.png'.format(home,multipole,gal,GC,rmin,rmax),bbox_tight=True)
plt.close()


# wp binning
smin=5; smax=30
binfilewp = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
selwp = (binfilewp['col3']<smax)&(binfilewp['col3']>=smin)
binswp  = np.unique(np.append(binfilewp['col1'][selwp],binfilewp['col2'][selwp]))
swp = binfilewp['col3'][selwp]
# plot the wp
fig = plt.figure(figsize=(7,8))
spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for zbin in range(zbinnum):
    for k in range(1):
        values=[np.zeros_like(OBSwps[zbin]),OBSwps[zbin]]
        err   = [np.ones_like(OBSwps[zbin]),errbarwps[zbin]]

        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(swp,(OBSwps[zbin]-values[j])/err[j],errbarwps[zbin]/err[j],color=colors[zbin], marker='^',ecolor=colors[zbin],ls="none",label='z{}z{}_pi80'.format(zmins[zbin],zmaxs[zbin]))
            ax[j,k].plot(swp,(wps[zbin][:,1]-values[j])/err[j],color=colors[zbin],label='_hidden')
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

plt.savefig('{}wp_bestfit_{}_{}_{}-{}Mpch-1_pi80.png'.format(home,gal,GC,smin,smax),bbox_tight=True)
plt.close()

# plot
pimaxs = [25,30,35]

for h,pimax in enumerate(pimaxs):
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    
    for zbin in range(zbinnum): 
        zmin,zmax = zmins[zbin],zmaxs[zbin]
        fileroot = '{}MCMCout/zbins_{}/{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,function,rscale,gal,GC,zmin,zmax)
        wp = np.loadtxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,pimax))

        # pimax files
        pythonsel = (wp[:,0]>smin)&(wp[:,0]<smax)
        wp = wp[pythonsel,:]
        # observations
        if (gal == 'LRG')|(gal=='ELG'):
            if rscale == 'linear':
                covfitswp  = '{}catalog/nersc_{}_{}_{}/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,'wp',gal,ver,gal,zmin,zmax,pimax) 
                obs2pcfwp  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}_pi{}.dat'.format(home,gal,ver,gal,GC,ver,pimax)
            elif rscale == 'log':
                covfitswp = '{}catalog/nersc_zbins_wp_mps_{}/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,gal,gal,zmin,zmax,pimax) 
                obs2pcfwp  = '{}catalog/nersc_zbins_wp_mps_{}/wp_rp_pip_eBOSS_{}_{}_{}_{}-{}_pi{}.dat'.format(home,gal,gal,GC,ver1,zmin,zmax,pimax)
            colwp   = 'col3'
        else:
            obs2pcfwp = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}_pi{}.wp'.format(home,gal,zmin,zmax,pimax)
            covfitswp = '{}catalog/BOSS_zbins_wp/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,gal,zmin,zmax,pimax)
            colwp   = 'col1'
        # observation rp selection
        obscfwp = Table.read(obs2pcfwp,format='ascii.no_header')
        selwp = (obscfwp[colwp]<smax)&(obscfwp[colwp]>=smin)
        OBSwp   = obscfwp['col4'][selwp]
        # Read the covariance matrices
        hdu = fits.open(covfitswp) 
        mockswp = hdu[1].data[GC+'mocks']
        Nmockwp = mockswp.shape[1] 
        errbarwp = np.std(mockswp,axis=1)
        hdu.close()  

        # 
        OBSwps[zbin] = OBSwp
        wps[zbin]     = wp
        errbarwps[zbin] = errbarwp

        for k in range(1):
            values=[np.zeros_like(OBSwp),OBSwp]
            err   = [np.ones_like(OBSwp),errbarwp]

            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k]);#import pdb;pdb.set_trace()
                ax[j,k].errorbar(swp,(OBSwp-values[j])/err[j],errbarwp/err[j],color=colors[zbin],marker='^',ecolor=colors[zbin],ls="none",label='z{}z{}_pi{}'.format(zmins[zbin],zmaxs[zbin],pimax))
                ax[j,k].plot(swp,(wp[:,1]-values[j])/err[j],color=colors[zbin],label='_hidden')
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

    plt.savefig('{}wp_bestfit_{}_{}_{}-{}Mpch-1_pi{}.png'.format(home,gal,GC,smin,smax,pimax),bbox_tight=True)
    plt.close()
"""
# plot the PDF
fig,ax = plt.subplots()
for zbin in range(zbinnum):
    plt.plot(bbins[:-1],pdfs[zbin],color=colors[zbin],label='z{}z{}'.format(zmins[zbin],zmaxs[zbin]))
plt.xlim(0,1500)
plt.ylim(1e-5,1.0)
plt.yscale('log')
plt.legend(loc=0)
plt.ylabel('prob of having a galaxy')
plt.xlabel('Vpeak (km/s)')
plt.savefig(home+'best_SHAM_PDF_hist_{}_{}_log.png'.format(gal,GC))
plt.close()

fig,ax = plt.subplots()
for zbin in range(zbinnum):
    plt.plot(bbins[:-1],pdfs[zbin],color=colors[zbin],\
        label='z{}z{}, PDF max: {} km/s'.format(zmins[zbin],zmaxs[zbin],(bbins[:-1])[pdfs[zbin]==max(pdfs[zbin][~np.isnan(pdfs[zbin])])]))
plt.xlim(0,1500)
plt.legend(loc=0)
plt.ylabel('prob of having a galaxy')
plt.xlabel('Vpeak (km/s)')
plt.savefig(home+'best_SHAM_PDF_hist_{}_{}.png'.format(gal,GC))
plt.close()
"""