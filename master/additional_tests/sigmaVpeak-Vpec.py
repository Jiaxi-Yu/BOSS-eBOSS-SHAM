import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from astropy.table import Table
from astropy.io import fits
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
gal      = 'ELG'
GC       = 'NGC'
date     = '0526' 
npoints  = 200#int(sys.argv[3])
nseed    = 2
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
fileroot = 'MCMCout/'+date+'/HAM_'+gal+'_'+GC+'/multinest_'

if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
    halofile = home+'catalog/UNIT_hlist_0.53780.fits.gz' 
    
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

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
halo = fits.open(halofile)
# make sure len(data) is even
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
halo.close()
datac = np.zeros((len(data),2))
datac[:,0] = np.sqrt(data['VX']**2+data['VY']**2+data['VZ']**2)
datac[:,1] = np.copy(data['Vpeak'])
half = int(len(data)/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 


# HAM application
def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(datac[:,-1])
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:LRGnum]] #3.06s
    LRGscat[:,-1]*=sigma_high
    return LRGscat
    
    
# plot the best-fit
parameters = ["sigma","vcut"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1]))))   

i=0
for xi in xi1_ELG:
    fig,ax = plt.subplots()
    plt.scatter(xi[:,1],xi[:,0],alpha=0.1,s=5)
    plt.ylabel('$V_{pec}$ (km/s)')
    plt.xlabel('$\sigma$$V_{peak}$ (km/s)')
    i+=1
    plt.savefig('sigmaVpeak-Vpec{}.png'.format(i))
    plt.close()

num=99
k=0
Vbins = np.linspace(20,500,num)
pecbins = np.zeros(num-1)
for xi in xi1_ELG:
    fig,ax = plt.subplots()
    for i in range(num-1):
        pecbins[i] = np.mean(xi[:,0][(xi[:,1]>Vbins[i])&(xi[:,1]<Vbins[i+1])])
    plt.scatter((Vbins[1:]+Vbins[:-1])/2,pecbins)
    plt.ylabel('$V_{pec}$ (km/s)')
    plt.xlabel('$\sigma$$V_{peak}$ (km/s)')
    k+=1
    plt.savefig('sigmaVpeak-Vpec_mean{}.png'.format(k))
    plt.close()
    