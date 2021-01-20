import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack
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
import matplotlib.gridspec as gridspec 
import sys
import pymultinest

# variables
#GC = 'NGC'#
GC       = sys.argv[1] #
date2    = '0810'
nseed    = 1
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
PDFmax   = 2000
home      = '/global/cscratch1/sd/jiaxi/master/'
fileroot1 = 'MCMCout/3-param_'+date2+'/LRG_'+GC+'/multinest_'
fileroot2 = 'MCMCout/3-param_'+date2+'/ELG_'+GC+'/multinest_'

# data for LRG and ELG
LRGnum1   = int(6.26e4)
zmin     = 0.6
zmax     = 1.0
z1 = 0.7018
halofile1 = home+'catalog/UNIT_hlist_0.58760.fits.gz' 

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
halo = fits.open(halofile1)
# make sure len(data) is even
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
halo.close()
print('1.selecting only the necessary variables...')
datac1 = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac1[:,i] = np.copy(data[key])
#V = np.copy(data[var]).astype('float32')
datac1 = datac1.astype('float32')
data = np.zeros(0)


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
def sham_tpcf(dat,uni,uni1,sigM,sigV,Mtrun):
    if dat==1:
        x00,x20,x40,n20=sham_cal(datac1,z1,LRGnum1,uni,sigM,sigV,Mtrun)
        x01,x21,x41,n21=sham_cal(datac1,z1,LRGnum1,uni1,sigM,sigV,Mtrun)
    if dat==2:
        x00,x20,x40,n20=sham_cal(datac2,z2,LRGnum2,uni,sigM,sigV,Mtrun)
        x01,x21,x41,n21=sham_cal(datac2,z2,LRGnum2,uni1,sigM,sigV,Mtrun)
    return [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2,(n20+n21)/2]

def sham_cal(DATAC,z,LRGnum,uniform,sigma_high,sigma,v_high):
    half = int(len(DATAC)/2)
    datav = DATAC[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = (DATAC[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]
    n,bins0=np.histogram(LRGscat[:,-1],bins=50,range=(0,PDFmax))

    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
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
    return [xi0_single,xi2_single,xi4_single,n]

# read the posterior file
parameters1 = ["sigma","Vsmear","Vceil"]
npar1 = len(parameters1)
a1 = pymultinest.Analyzer(npar1, outputfiles_basename = fileroot1)

parameters2 = ["sigma","Vsmear","Vceil"]
npar2 = len(parameters2)
a2 = pymultinest.Analyzer(npar2, outputfiles_basename = fileroot2)

# plot the best-fit
# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac1)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac1)).astype('float32') for x in range(nseed)] 
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(repeat(1),uniform_randoms,uniform_randoms1,repeat(np.float32(a1.get_best_fit()['parameters'][0])),repeat(np.float32(a1.get_best_fit()['parameters'][1])),repeat(np.float32(a1.get_best_fit()['parameters'][2])))) 
uniform_randoms=[]
uniform_randoms1=[]
    
#ELG
# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac2)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac2)).astype('float32') for x in range(nseed)] 
with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(repeat(2),uniform_randoms,uniform_randoms1,repeat(np.float32(a2.get_best_fit()['parameters'][0])),repeat(np.float32(a2.get_best_fit()['parameters'][1])),repeat(np.float32(a2.get_best_fit()['parameters'][2]))))    


nlist = [xi_ELG[x][3] for x in range(nseed)]
narray = np.array(nlist).T
n1list = [xi1_ELG[x][3] for x in range(nseed)]
n1array = np.array(n1list).T

# plot the galaxy probability distribution and the real galaxy number distribution 
n1,bins1=np.histogram(datac1[:,-1],bins=50,range=(0,PDFmax))
n2,bins2=np.histogram(datac2[:,-1],bins=50,range=(0,PDFmax))
fig =plt.figure(figsize=(20,6))
ax = plt.subplot2grid((1,3),(0,1))
binmid = (bins1[:-1]+bins1[1:])/2
ax.errorbar(bins1[:-1],np.mean(narray,axis=-1)/LRGnum1,yerr = np.std(narray,axis=-1)/LRGnum1,color='m',alpha=0.7,ecolor='m',label='LRG-{}'.format(GC),ds='steps-mid')
ax.errorbar(binmid,np.mean(n1array,axis=-1)/LRGnum2,yerr=np.std(n1array,axis=-1)/LRGnum2,color='c',alpha=0.7,ecolor='c',label='ELG-{}'.format(GC),ds='steps-mid')
plt.ylabel('$V_{peak}$ frequency')
plt.legend(loc=2)
plt.title('Vpeak histogram in {} '.format(GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(PDFmax,10)

ax = plt.subplot2grid((1,3),(0,0))
ax.errorbar(bins1[:-1],np.mean(narray,axis=-1)/(n1),yerr = np.std(narray,axis=-1)/(n1),color='m',alpha=0.7,ecolor='m',label='LRG-{}'.format(GC),ds='steps-mid')
ax.errorbar(binmid,np.mean(n1array,axis=-1)/(n2),yerr=np.std(n1array,axis=-1)/(n2),color='c',alpha=0.7,ecolor='c',label='ELG-{}'.format(GC),ds='steps-mid')
plt.ylabel('prob. to have 1 galaxy in 1 halo')
plt.legend(loc=1)
plt.title('Vpeak probability distribution in {} '.format(GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(PDFmax,10)

ax = plt.subplot2grid((1,3),(0,2))
ax.errorbar(bins1[:-1],np.mean(narray,axis=-1),yerr = np.std(narray,axis=-1),color='m',alpha=0.7,ecolor='m',label='LRG-{}'.format(GC),ds='steps-mid')
ax.step(binmid,n1,color='k',label='UNIT z1={:.3}'.format(z1))
ax.errorbar(binmid,np.mean(n1array,axis=-1),yerr=np.std(n1array,axis=-1),color='c',alpha=0.7,ecolor='c',label='ELG-{}'.format(GC),ds='steps-mid')
ax.step(binmid,n2,color='k',alpha = 0.5,label='UNIT z2={:.3} '.format(z2))
plt.yscale('log')
plt.ylabel('galaxy numbers')
plt.legend(loc=2)
plt.title('Vpeak distribution in {}'.format(GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(PDFmax,10)

plt.savefig('LRGvsELG_'+GC+'.png',bbox_tight=True)
plt.close()
