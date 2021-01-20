import time
initial = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
import warnings
from multiprocessing import Pool 
from itertools import repeat
import glob
import sys

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
date     = '0905'#'0810' 
npoints  = 150 
nseed    = 10
rscale   = 'linear' # 'log'
multipoles= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
function = 'mps' # 'wp'
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
fileroot = 'MCMCout/3-param_'+date+'/'+gal+'_'+GC+'/multinest_'


if gal == 'LRG':
    LRGnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    ver='v7_2'
    halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 

if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    ver='v7'
    halofile = home+'catalog/UNIT_hlist_0.53780.fits.gz'

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate linear bins or read log bins
if rscale=='linear':
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
    #obs2pcf = home+'catalog/nersc_mps_'+gal+'_'+ver+'/2PCF_'+function+'_'+rscale+'_'+gal+'_'+GC+'.dat'
    #covfits  = home+'catalog/nersc_mps_'+gal+'_'+ver+'/2PCF_'+function+'_'+rscale+'_'+gal+'_mocks_'+multipoles+'.fits.gz'
    #COVR = np.loadtxt(home+'catalog/nersc_mps_'+gal+'_'+ver+'/covR-'+gal+'_'+GC+'-5_25-'+multipoles+'.dat')
    obs2pcf = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
    COVR  = np.loadtxt(home+'2PCF/obs/covR_'+gal+'_'+GC+'.dat')
    covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_quad.fits.gz'
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) # cov([mono,quadru])
    #mocks = hdu[1].data[GC+'mocks']
    mocks = hdu[1].data[multipoles]
    Nmock =mocks.shape[1]
    hdu.close()
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
    print('the covariance matrix and the observation 2pcf vector are ready.')


# SHAM halo catalogue
print('reading the halo catalogue and selecting only the necessary variables...')
halo = fits.open(halofile)
length  =len(halo[1].data)
# make sure len(data) is even
if length%2==1:
    datac = np.zeros((length-1,5))
    for i,key in enumerate(['X','Y','Z','VZ',var]):
        datac[:,i] = np.copy(halo[1].data[key][:-1])
    datac = datac.astype('float32')
else:
    datac = np.zeros((length,5))
    for i,key in enumerate(['X','Y','Z','VZ',var]):
        datac[:,i] = np.copy(halo[1].data[key])
    datac = datac.astype('float32')
halo.close()

half = int32(length/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    x00,x20,x40=sham_cal(uni,sigM,sigV,Mtrun)
    x01,x21,x41=sham_cal(uni1,sigM,sigV,Mtrun)
    return [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2]

def sham_cal(uniform,sigma_high,sigma,v_high):
    datav = datac[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]
    
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
    z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    z_redshift %=boxsize
    
    # calculate the 2pcf of the SHAM galaxies
    if function =='mps':
        # count the galaxy pairs and normalise them
        DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
        # calculate the 2pcf and the multipoles
        mono = (DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1)
        quad = mono * 2.5 * (3 * mu**2 - 1)
        hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        # use sum to integrate over mu 
 
    return [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu]
    

# chi2
def chi2(sigma_M,sigma_V,M_ceil):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(float32(sigma_M)),repeat(float32(sigma_V)),repeat(float32(M_ceil))))

         # average the result for multiple seeds
        xi0,xi2,xi4 = mean(xi0_tmp,axis=0,dtype='float32')[0],mean(xi0_tmp,axis=0,dtype='float32')[1],mean(xi0_tmp,axis=0,dtype='float32')[2]

        # identify the fitting multipoles
    model = append(xi0,xi2)
    Ns = int(mocks.shape[0]/2)
    mock = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:]))
    covcut  = cov(mock).astype('float32')
    #OBS   = append(obscf['col4'],obscf['col5']).astype('float32')  
    OBS   = append(obscf['col3'],obscf['col4']).astype('float32')  

    # calculate the covariance, residuals and chi2
    Nbins = len(OBS)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    return [res.dot(covR.dot(res)),res.dot(COVR.dot(res))]
    
print(chi2(sys.argv[3],sys.argv[4],sys.argv[5]))
