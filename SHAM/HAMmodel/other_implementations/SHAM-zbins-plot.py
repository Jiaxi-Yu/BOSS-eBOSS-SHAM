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
function = sys.argv[4] #'mps' # 'wp'
zmin     = sys.argv[5]
zmax     = sys.argv[6]
finish   = int(sys.argv[7])
nseed    = 10
date     = '0122'
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 32
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home     = '/home/astro/jiayu/Desktop/SHAM/'
fileroot = '{}MCMCout/zbins_{}/{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,function,rscale,gal,GC,zmin,zmax)
bestfit   = '{}bestfit_{}_{}.dat'.format(fileroot[:-10],function,date)
cols = ['col4','col5']

# read the posterior file
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

# getdist results
sample = loadMCSamples(fileroot)
print('chain Results:')
stats = sample.getMargeStats()
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
sigma = np.zeros(npar)
for i in range(npar):
    par = stats.parWithName(parameters[i])
    mean[i] = par.mean
    sigma[i] = par.err
    lower[i] = par.limits[0].lower
    upper[i] = par.limits[0].upper
    best[i] = (lower[i] + upper[i]) * 0.5
    print('getdist {0:s}: [{1:.6f}, {2:.6f}]'.format( \
        parameters[i],  lower[i], upper[i]))
#getdist plot
plt.rcParams['text.usetex'] = False
g = plots.getSinglePlotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g = plots.getSubplotPlotter()
g.triangle_plot(sample,parameters, filled=True)
g.export('{}{}_{}_{}_posterior.png'.format(fileroot[:-10],date,gal,GC))
plt.close()

# write the report
if finish:
    stats = a.get_stats()    
    file = fileroot[:-10]+gal+'_'+GC+'_Vzsmear_report.txt'
    f = open(file,'a')
    f.write('{} {} multinest: \n'.format(gal,GC))
    f.write('(-2)* max loglike: {} \n'.format(-2*a.get_best_fit()['log_likelihood']))
    f.write('max-loglike params: {}\n'.format(a.get_best_fit()['parameters']))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('getdist 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
    f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s,Vceil {:.6}+{:.6}{:.6}km/s  \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2],upper[2]-a.get_best_fit()['parameters'][2],lower[2]-a.get_best_fit()['parameters'][2]))

    for j in range(npar):
        lower[j], upper[j] = stats['marginals'][j]['1sigma']
        print('getdist {0:s}: [{1:.6f} {2:.6f}]'.format(parameters[j],  upper[j], lower[j]))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('multinest analyser results: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
    f.write('another way around: sigma {0:.6}+{1:.6}{2:.6}, sigma_smear {3:.6}+{4:.6}{5:.6}km/s,Vceil {6:.6}+{7:.6}{8:.6}km/s  \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2],upper[2]-a.get_best_fit()['parameters'][2],lower[2]-a.get_best_fit()['parameters'][2]))
    f.close()

# cormer results
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
plt.savefig(fileroot[:-10]+gal+'_'+GC+'_posterior_check.png')
print('the best-fit parameters: sigma {:.4},Vsmear {:.6} km/s, Vceil {:.6} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2]))
print('its chi2: {:.6}'.format(-2*a.get_best_fit()['log_likelihood']))

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
    if function =='mps': 
        binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
    elif function =='wp':
        binfile = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
        
    bins  = np.unique(np.append(binfile['col1'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)],binfile['col2'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]))
    s = binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]
    nbins = len(bins)-1
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    
    if function == 'mps':
        if gal == 'LRG':
            ver = 'v7_2'
            extra = binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]**2
        else:
            ver = 'v7'
            extra = np.ones(binmax-binmin)
        # filenames
        covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
        obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
    elif function == 'wp':
        if gal == 'LRG':
            ver = 'v7_2'
        else:
            ver = 'v7'
        covfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(home,function,gal,ver,function,gal) 
        obs2pcf  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver)
    
    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1]
    hdu.close()
    # observations
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance and errobar for chi2
    if function == 'mps':
        Ns = int(mocks.shape[0]/2)
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col4']/extra,obscf['col5']/extra).astype('float32')# LRG columns are s**2*xi
    elif function == 'wp':
        mocks = hdu[1].data[GC+'mocks'][binmin:binmax,:]
        covcut  = cov(mocks).astype('float32') 
        OBS   = np.array(obscf['col4']).astype('float32')
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

# analytical RR
mu = (np.linspace(0,mu_max,nmu+1)[:-1]+np.linspace(0,mu_max,nmu+1)[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))
# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# SHAM halo catalogue
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
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:])))/sigma_high #0.5s
    # modified Vpeak_scat
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],SHAMnum)[:(SHAMnum)]]
    #LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    #datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    #LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    #datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    
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

# plot the results
errbar = np.std(mocks,axis=1)
if function =='mps':
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

    plt.savefig('{}cf_{}_bestfit_{}_{}_{}-{}Mpch-1.png'.format(fileroot[:-10],multipole,gal,GC,rmin,rmax),bbox_tight=True)
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
    plt.savefig('{}{}_bestfit_{}_{}_fit{}-{}Mpch-1.png'.format(fileroot[:-10],function,gal,GC,rmin,rmax),bbox_tight=True)
    plt.close()