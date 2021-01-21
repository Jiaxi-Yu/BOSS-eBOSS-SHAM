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
gal      = 'ELG'
GC       = 'SGC'
date     = '1118'#'1027'#'1211'#'1118'
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
function = 'mps'#'mps'#'wp'
home      = '/global/cscratch1/sd/jiaxi/SHAM/'
direc    = '/global/homes/j/jiaxi/'
bestfit   = 'bestfit_{}_{}.dat'.format(function,date)

# read the posterior file
parameters2 = ["sigma","Vsmear","Vceil"]
npar2 = len(parameters2)
fileroot2 = '{}MCMCout/indexcut_{}/LRG_SGC_hexa_prior3_2-35/multinest_'.format(direc,date)
#'MCMCout/indexcut_{}/mps_ELG_SGC_5-35/multinest_'.format(date)
# LRG_SGC_sigma_prior1, LRG_SGC_sigma_prior
a = pymultinest.Analyzer(npar2, outputfiles_basename = fileroot2)

# plot the posterior
A=a.get_equal_weighted_posterior()
figure = corner.corner(A[:,:3],labels=[r"$sigma$",r"$Vsmear$", r"$Vceil$"],\
                       show_titles=True,title_fmt=None)
axes = np.array(figure.axes).reshape((3,3))
for yi in range(3): 
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(a.get_best_fit()['parameters'][xi], color="g")
        ax.axhline(a.get_best_fit()['parameters'][yi], color="g")
        ax.plot(a.get_best_fit()['parameters'][xi],a.get_best_fit()['parameters'][yi], "sg") 
plt.savefig(gal+'_'+GC+'_posterior_check.png')
print('the best-fit parameters: sigma {:.4},Vsmear {:.6} km/s, Vceil {:.6} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2]))
print('its chi2: {:.6}'.format(-2*a.get_best_fit()['log_likelihood']))

# getdist plot
sample = loadMCSamples(fileroot2)
plt.rcParams['text.usetex'] = False
g = plots.getSinglePlotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g = plots.getSubplotPlotter()
plt.clabel(g, inline=True, fontsize=8)
g.triangle_plot(sample,parameters2, filled=True)#,title_limit=1)
g.export('{}_{}_{}_posterior.png'.format(date,gal,GC))
plt.close()

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

if (rscale=='linear')&(function=='mps'):
    # generate s bins
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
    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(direc,gal,ver,function,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(direc,gal,ver,function,rscale,gal,multipole)
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    #mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:],mock[binmin+200*2:binmax+200*2,:]))
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:]))
    covcut  = cov(mocks).astype('float32')
    errbar = np.std(mocks,axis=1)
    hdu.close()
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
    #OBS   =hstack((obscf['col4'],obscf['col5'],obscf['col6'])).astype('float32')
    if gal == 'LRG':
        cols = ['col4','col5']
    else:
        cols = ['col3','col4']
    OBS   = append(obscf[cols[0]],obscf[cols[1]]).astype('float32')
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    print('the covariance matrix and the observation 2pcf vector are ready.')
elif (rscale=='log')&(function=='mps'):
    """
    if gal=='ELG':
        binfile = Table.read(home+'cheng_HOD_{}/mps_log_{}_NGC+SGC_eBOSS_v7_zs_0.70-0.90.dat'.format(gal,gal),format='ascii.no_header')
    else:
        binfile = Table.read(home+'cheng_HOD_{}/mps_log_{}_NGC+SGC_eBOSS_v7_2_zs_0.60-0.80.dat'.format(gal,gal),format='ascii.no_header')   

    bins  = np.unique(np.append(binfile['col1'],binfile['col2']))
    bins = bins[bins<rmax]
    nbins = len(bins)-1
    """
elif function=='wp':
    # the log binned wp
    covfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(direc,function,gal,ver,function,gal) 
    obs2pcf  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(direc,gal,ver,gal,GC,ver)
    # bin
    binfile = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    # observation
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf = obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
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
else:
    print('wrong 2pcf function input')

# SHAM halo catalogue, keep them even
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
print('read the halo catalogue costs {}s'.format(time.time()-read))

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    if function == 'mps':
        x00,x20,v0= sham_cal(uni,sigM,sigV,Mtrun)
        x01,x21,v1= sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = [(x00+x01)/2,(x20+x21)/2,(v0+v1)/2]
    else:        
        x00,v0    = sham_cal(uni,sigM,sigV,Mtrun)
        x01,v1    = sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = [(x00+x01)/2,(v0+v1)/2]
    return tpcf

def sham_cal(uniform,sigma_high,sigma,v_high):
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]        
    datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
    z_redshift  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    z_redshift %=boxsize
    
    if function == 'mps':
        DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
        # calculate the 2pcf and the multipoles
        mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
        quad = mono * 2.5 * (3 * mu**2 - 1)
        # use sum to integrate over mu
        SHAM_array = [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,min(datav)]
    else:
        wp_dat = wp(boxsize,80,nthread,bins,LRGscat[:,2],LRGscat[:,3],z_redshift)
        SHAM_array = [wp_dat['wp'],min(datav)]
    return SHAM_array

# calculate the SHAM 2PCF
with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2]))))) 

if function =='mps':
    np.savetxt(bestfit,np.vstack((np.mean(xi1_ELG,axis=0)[0],np.mean(xi1_ELG,axis=0)[1])).T)
    print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[2]))

    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(cols,[int(0),int(200)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)[k]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[k*nbins:(k+1)*nbins],color='k', marker='o',ecolor='k',ls="none")
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                label = ['SHAM','PIP obs 1$\sigma$']#['SHAM','SHAM_2nd']#,'PIP obs 1$\sigma$']#,'CP obs 1$\sigma$']
                if k==0:
                    plt.legend(label,loc=2)
                else:
                    plt.legend(label,loc=1)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))#('\Delta\\xi_{}$'.format(k*2))#

    plt.savefig('cf_{}_bestfit_{}_{}_{}-{}Mpch-1.png'.format(multipole,gal,GC,rmin,rmax),bbox_tight=True)
    plt.close()
else:
    np.savetxt(bestfit,np.mean(xi1_ELG,axis=0)[0])
    print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[1]))
    
    fig = plt.figure(figsize=(5,6))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,1), dtype=type(plt.axes))
    k=0
    values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)[0]]
    for j in range(2):
        ax[j,k] = fig.add_subplot(spec[j,k])
        ax[j,k].plot(s,(np.mean(xi1_ELG,axis=0)[0]-values[j]),c='c',alpha=0.6)
        ax[j,k].errorbar(s,(OBS-values[j]),errbar,color='k', marker='o',ecolor='k',ls="none",markersize = 4)
        plt.xlabel('$r_p$ (Mpc $h^{-1}$)')
        plt.xscale('log')
        if (j==0):
            ax[j,k].set_ylabel('wp')
            label = ['SHAM','PIP obs 1$\sigma$']
            plt.legend(label,loc=0)
            plt.title('projected 2-point correlation function: {} in {}'.format(gal,GC))
        if (j==1):
            ax[j,k].set_ylabel('$\Delta$ wp')
    plt.savefig('{}_bestfit_{}_{}_fit{}-{}Mpch-1.png'.format(function,gal,GC,rmin,rmax),bbox_tight=True)
    plt.close()
