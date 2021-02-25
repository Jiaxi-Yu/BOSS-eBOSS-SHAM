import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack,hstack
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
import getdist
import sys
import pymultinest
import h5py

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
rscale   = sys.argv[3] #'linear' # 'log'
function = sys.argv[4] #'mps' # 'wp'
zmin     = sys.argv[5]
zmax     = sys.argv[6]
mode     = sys.argv[7]
nseed    = 20
date     = '0220'
npoints  = 100 
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
home     = '/global/cscratch1/sd/jiaxi/SHAM/'
direc    = '/global/homes/j/jiaxi/'
fileroot = '{}MCMCout/percentcut_{}/{}_{}_{}_{}_z{}z{}/multinest_'.format(direc,date,function,rscale,gal,GC,zmin,zmax)


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
       
    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(direc,gal,ver,function,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(direc,gal,ver,function,rscale,gal,multipole)
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
        covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(direc,gal,function,rscale,gal,zmin,zmax,multipole) 
        obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(direc,gal,function,rscale,gal,GC,ver,zmin,zmax)
    elif function == 'wp':
        if gal == 'LRG':
            ver = 'v7_2'
        else:
            ver = 'v7'
        covfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(direc,function,gal,ver,function,gal) 
        obs2pcf  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(direc,gal,ver,gal,GC,ver)
    
    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1] 
    errbar = np.std(mocks,axis=1)
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
        x00,x20= sham_cal(uni,sigM,sigV,Mtrun)
        x01,x21= sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = [(x00+x01)/2,(x20+x21)/2]
    else:        
        x00    = sham_cal(uni,sigM,sigV,Mtrun)
        x01    = sham_cal(uni1,sigM,sigV,Mtrun)
        tpcf   = (x00+x01)/2
    return tpcf

def sham_cal(uniform,sigma_high,sigma,v_high):
    """
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
    LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
    """
    datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = datac[argpartition(-datav,SHAMnum+int(len(datac)*v_high/100))[:(SHAMnum+int(len(datac)*v_high/100))]]
    datav = datav[argpartition(-datav,SHAMnum+int(len(datac)*v_high/100))[:(SHAMnum+int(len(datac)*v_high/100))]]
    LRGscat = LRGscat[argpartition(-datav,int(len(datac)*v_high/100))[int(len(datac)*v_high/100):]]


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
        SHAM_array = [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu]
    else:
        wp_dat = wp(boxsize,80,nthread,bins,LRGscat[:,2],LRGscat[:,3],z_redshift)
        SHAM_array = wp_dat['wp']
    return SHAM_array

# chi2
def chi2(sigma_M,sigma_V,M_ceil):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(float32(sigma_M)),repeat(float32(sigma_V)),repeat(M_ceil)))
    
    if function == 'mps':
        # average the result for multiple seeds
        xi0,xi2 = mean(xi0_tmp,axis=0,dtype='float32')[0],\
                  mean(xi0_tmp,axis=0,dtype='float32')[1]
        model = append(xi0,xi2)
    else:
        model = mean(xi0_tmp,axis=0,dtype='float32')
    # calculate the residuals and chi2
    res = OBS-model
    if mode == 'debug':
        return [res.dot(covR.dot(res)),model]
    else:
        return res.dot(covR.dot(res))

# prior
def prior(cube, ndim, nparams):
    if (gal=='LRG'):
        # the same as LRG_SGC_4-n
        cube[0] = 2.5*cube[0]
        cube[1] = 100*cube[1]+60
        #cube[2] = 2.0*cube[2]+4.0 
        cube[2] = 0.09*cube[2]+0.01
    elif gal=='ELG':
        cube[0] = 3*cube[0]+0.5
        cube[1] = 60*cube[1]      
        #cube[2] = 2.0*cube[2]+5.0 
        cube[2] = 9.0*cube[2]+1.0

# loglikelihood = -0.5*chi2    
def loglike(cube, ndim, nparams):
    sigma_high,sigma,vhigh = cube[0],cube[1],cube[2]
    return -0.5*chi2(sigma_high,sigma,vhigh)   

# number of dimensions our problem has
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
if os.path.exists(fileroot[:-10])==False:
    os.mkdir(fileroot[:-10])
    
if mode == 'debug':
    print('debug mode on')
    chisq, xis = chi2(0.5,100.,0.05)
    print('chi2 = {:.3f}'.format(chisq))
    np.savetxt('percent_cut-python.dat',xis.reshape(2,nbins).T,header='xi0 xi2')
else:
    # run MultiNest & write the parameter's name
    pymultinest.run(loglike, prior, npar,n_live_points= npoints, outputfiles_basename=fileroot, resume =True, verbose = True,n_iter_before_update=5,write_output=True)

    # results
    fin = time.time()  
    file = fileroot[:-10]+gal+'_'+GC+'_Vzsmear_report.txt'
    f = open(file,'a')
    f.write('the total {} in {} SHAM costs {:.6} s in 16 cores \n'.format(gal,GC,fin-init))
    f.close()
