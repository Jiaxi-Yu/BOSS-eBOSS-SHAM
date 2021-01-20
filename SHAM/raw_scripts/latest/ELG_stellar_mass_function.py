import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,vstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec 
import sys
import pymultinest

# variables
date2    = '0810'
nseed    = 10
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
scatrange = [2.573,2.588,2.526,2.540] 
scatnum = 50

# data for ELG
LRGnum2   = int(2.93e5)
zmin     = 0.6
zmax     = 1.1
z2 = 0.8594
halofile2 = home+'catalog/UNIT_hlist_0.53780.fits.gz' 

# cosmological parameters
Ode = 1-Om
# generate separation bins
if rscale=='linear':
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

# create the halo catalogue and plot their 2pcf
print('reading the halo catalogue for creating the galaxy catalogue...')
# make sure len(data) is even
halo = fits.open(halofile2)
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
halo.close()
print('2. selecting only the necessary variables...')
datac2 = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac2[:,i] = np.copy(data[key])
#V = np.copy(data[var]).astype('float32')
datac2 = datac2.astype('float32')
data = np.zeros(0)

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    x00,x20,x40,Vpeak_sel0,Vpeak_scat0=sham_cal(datac2,z2,LRGnum2,uni,sigM,sigV,Mtrun)
    x01,x21,x41,Vpeak_sel1,Vpeak_scat1=sham_cal(datac2,z2,LRGnum2,uni1,sigM,sigV,Mtrun)
    return [vstack((x00,x01)),vstack((x20,x21)),vstack((x40,x41)),vstack((Vpeak_sel0,Vpeak_sel1)),vstack((Vpeak_scat0,Vpeak_scat1))]

def sham_cal(DATAC,z,LRGnum,uniform,sigma_high,sigma,v_high):
    half = int(len(DATAC)/2)
    datav = DATAC[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = (DATAC[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:LRGnum]]

    # transfer to the redshift space
    scathalf = int(LRGnum/2)
    H = 100*np.sqrt(Om*(1+z)**3+Ode)
    z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
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
    print('calculation finish')
    return [xi0_single,xi2_single,xi4_single,LRGscat[:,-1],(datav[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]]

# read the posterior file
fileroot1 = 'MCMCout/3-param_'+date2+'/ELG_SGC/multinest_'
parameters1 = ["sigma","Vsmear","Vceil"]
npar1 = len(parameters1)
a1 = pymultinest.Analyzer(npar1, outputfiles_basename = fileroot1)

fileroot2 = 'MCMCout/3-param_'+date2+'/ELG_NGC/multinest_'
parameters2 = ["sigma","Vsmear","Vceil"]
npar2 = len(parameters2)
a2 = pymultinest.Analyzer(npar2, outputfiles_basename = fileroot2)

# plot the best-fit for ELGs
# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac2)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac2)).astype('float32') for x in range(nseed)] 


with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a1.get_best_fit()['parameters'][0])),repeat(np.float32(a1.get_best_fit()['parameters'][1])),repeat(np.float32(a1.get_best_fit()['parameters'][2])))) 

with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a2.get_best_fit()['parameters'][0])),repeat(np.float32(a2.get_best_fit()['parameters'][1])),repeat(np.float32(a2.get_best_fit()['parameters'][2]))))  


# Vpeak, scattered Vpeak and stellar mass
# dictionary initialisation
Vpeak_stat = {}
for l,keys in enumerate(['NGC-Vpeak','NGC-Vpeak_scat']):
    Vpeak_stat[keys] = (np.array([xi_ELG[x][3+l] for x in range(nseed)]).T).reshape(LRGnum2,nseed*2)
for l,keys in enumerate(['SGC-Vpeak','SGC-Vpeak_scat']):
    Vpeak_stat[keys] = (np.array([xi1_ELG[x][3+l] for x in range(nseed)]).T).reshape(LRGnum2,nseed*2)


Nmedian,Nstd  = np.zeros(scatnum),np.zeros(scatnum)
# abundance matching in NGC and SGC
for q,GC in enumerate(['NGC','SGC']):
    # dictionary array initialisation
    Vpeak_stat[GC+'-Vpeak_scat_sort'] = np.zeros_like(Vpeak_stat[GC+'-Vpeak_scat'])
    Vpeak_stat[GC+'-Vpeak_sort'] = np.zeros_like(Vpeak_stat[GC+'-Vpeak'])
    
    # stellar mass readong
    file = '/global/cscratch1/sd/jiaxi/master/catalog/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits'
    hdu = fits.open(file)
    data = hdu[1].data['fast_lmass']
    
    # stellar mass random up/down-sampling in order to have the same number as SHAM galaxies
    data_upsample = np.random.choice(data,LRGnum2)
    
    # stellar mass descending sort to do the abundance matching
    Vpeak_stat[GC+'-Mstellar'] = -np.sort(-data_upsample)
    
    # sort scattered Vpeak and Vpeak according to scattered Vpeak
    Vpeak_stat[GC+'-Vpeak_scat_sort'] = np.sort(Vpeak_stat[GC+'-Vpeak_scat'],axis=0) 
    Vpeak_stat[GC+'-Vpeak_sort'] = np.take_along_axis(Vpeak_stat[GC+'-Vpeak'],np.argsort(Vpeak_stat[GC+'-Vpeak_scat'],axis=0),axis=0)
    # plot
    for k in range(2):
        fig,ax = plt.subplots()
        a,b,c,scat_M = ax.hist2d(np.log10(Vpeak_stat[GC+'-Vpeak_scat_sort'][:,k]),Vpeak_stat[GC+'-Mstellar'],bins=[scatnum,scatnum],range=[[scatrange[2*q],scatrange[2*q+1]],[9.5,11.5]],cmap='Blues')
        fig.colorbar(scat_M)
        plt.xlabel('log($V_{peak}^{scat}$)')
        plt.ylabel('log($M_*$)')
        plt.savefig('scattered_Vpeak-Mstellar_{}_{}.png'.format(GC,k))
        plt.close()
    
    for k in range(2):
        fig,ax = plt.subplots()
        a,b,c,org_M =ax.hist2d(np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k]),Vpeak_stat[GC+'-Mstellar']-np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k]),bins=[scatnum,scatnum],range=[[1.9,2.6],[7,9]],cmap='Blues')
        fig.colorbar(org_M)
        for j in range(scatnum-1):
            Nmedian[j] =np.median((Vpeak_stat[GC+'-Mstellar']-np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k]))[(np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k])>b[j])&(np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k])<=b[j+1])])
            Nstd = np.std((Vpeak_stat[GC+'-Mstellar']-np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k]))[(np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k])>b[j])&(np.log10(Vpeak_stat[GC+'-Vpeak_sort'][:,k])<b[j+1])])
        plt.errorbar((b[1:]+b[:-1])/2,Nmedian,Nstd,color='k',ecolor='k',ls="none",marker='o',lw=1,markersize=3)
        plt.xlabel('log($V_{peak}$)')
        plt.ylabel('log($M_*$)-log($V_{peak}$)')
        plt.xlim(1.9,2.6)
        plt.savefig('Vpeak-Mstellar_Vpeak_ratio_{}_{}.png'.format(GC,k))
        plt.close()
    
