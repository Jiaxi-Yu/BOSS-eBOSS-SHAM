import matplotlib 
matplotlib.use('agg')
import time
initial = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack
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

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
rscale   = sys.argv[3] #'linear' # 'log'
function = sys.argv[4] #'mps' # 'wp'
zmin     = sys.argv[5]
zmax     = sys.argv[6]

date     = '0905'#'0810' 
npoints  = 150 
nseed    = 30
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 1
rmax     = 30
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home      = '/global/cscratch1/sd/jiaxi/master/'
fileroot = 'MCMCout/3-param_'+date+'/'+gal+'_'+GC+'/multinest_'

if rscale =='linear':
    if gal == 'LRG':
        LRGnum   = int(6.26e4)
        z = 0.7018
        a_t = '0.58760'
        version = 'v7_2'
    else:
        LRGnum   = int(2.93e5)
        z = 0.8594
        a_t = '0.53780'
        version = 'v7'
        
    ================================================================
    # covariance matrices and observations    
    "the files should be ready"
    obsname = home+'catalog/nersc_mps_'+gal+'_'+version+'/'+function+'_'+rscale+'_'+gal+'_'+GC+'_pip_eBOSS_'+version+'_zs_'+zmin+'-'+zmax+'.dat'
    #obsname = home+'catalog/nersc_mps_'+gal+'_'+version+'/pair_counts_s-mu_pip_eBOSS_'+function+''+gal+'_'+GC+'_'+version+'.dat'
    
    "names to be revised": with function,rscale information
    covfits = home+'obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz' 
    obs2pcf  = home+'obs/'+gal+'_'+GC+'.dat'
    
    "the column name should be 'covariance' instead of multipole"
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[multipole]
    Nmock = mocks.shape[1] 
    errbar = np.std(mocks,axis=1)
    hdu.close()
    ==============================================================
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    if function == 'mps':
        # generate s bins
        bins  = np.arange(rmin,rmax+1,1)
        nbins = len(bins)-1
        binmin = rmin
        binmax = rmax
        # generate mu bins   
        s = (bins[:-1]+bins[1:])/2
        obscf = obscf[binmin:binmax]
    else:
        # read s bins
        bins  = np.unique(np.append(obscf['col1'],obscf['col2']))
        bins = bins[bins<rmax]
        obscf= obscf[bins<rmax]
        nbins = len(bins)-1
        s = obscf['col3']
    print('the covariance matrix and the observation 2pcf vector are ready.')
else:
    obsname = home+'catalog/nersc_zbins_wp_mps_'+gal+'/'+function+'_'+rscale+'_'+gal+'_'+GC+'_eBOSS_'+version+'_zs_'+zmin+'-'+zmax+'.dat'
    ================================================================
    "to be calculated":with gal, GC, function,rscale information
    covfits = home+'obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz' 
    obs2pcf  = home+'obs/'+gal+'_'+GC+'.dat'
    "the column name should be 'covariance' instead of multipole"
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[multipole]
    Nmock = mocks.shape[1] 
    errbar = np.std(mocks,axis=1)
    hdu.close()
    ==============================================================
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    # read s bins
    bins  = np.unique(np.append(obscf['col1'],obscf['col2']))
    bins = bins[bins<rmax]
    obscf= obscf[bins<rmax]
    nbins = len(bins)-1
    s = obscf['col3']
    
    if zmin=='0.60':
        ======================================================================
        if gal=='ELG':
            LRGnum = 
            z = # To be calculated
        else:
            LRGnum = 
            z=
        a_t = 
    elif zmin=='0.70':
        if gal=='ELG':
            LRGnum = 
            z = # To be calculated
        else:
            LRGnum = 
            z=
        a_t = 
    else:
        if gal=='ELG':
            LRGnum = 
            z = # To be calculated
        else:
            LRGnum = 
            z=
        a_t = 
        =====================================================================

if function == 'mps':
    mu = (np.linspace(0,mu_max,nmu+1)[:-1]+np.linspace(0,mu_max,nmu+1)[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))
    # Analytical RR calculation
    RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
    rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
    print('the analytical random pair counts are ready.')
    ====================================================================
    "2PCF obs: smin smax smid xi0 xi2 xi4"
    ====================================================================
    # preprocess the covariance matrix
    if multipole=='mono':
        mocks = mocks[binmin:binmax,:]
        covcut = cov(mocks).astype('float32')
        OBS   = obscf['col4'].astype('float32')
    elif multipole=='quad':
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+200:binmax+200,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')  
    else:
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+200:binmax+200,:],mocks[binmin+400:binmax+400,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col4'],obscf['col5'],obscf['col6']).astype('float32')

        
        # SHAM halo catalogue    
halofile = home+'catalog/UNIT_hlist_'+a_t+'.fits.gz'        

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

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
scathalf = int(LRGnum/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    if function == 'mps':
        x00,x20,x40=sham_cal(uni,sigM,sigV,Mtrun)
        x01,x21,x41=sham_cal(uni1,sigM,sigV,Mtrun)
        avg_result = [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2]
    else:
        wp0 = sham_cal(uni,sigM,sigV,Mtrun)
        wp1 = sham_cal(uni1,sigM,sigV,Mtrun)
        avg_result = (wp0+wp1)/2
    return avg_result

def sham_cal(uniform,sigma_high,sigma,v_high):
    datav = datac[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]
    
    # transfer to the redshift space
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
        SHAM_array = [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu]
    else:
        wp_dat = wp(boxsize,80,nthreads,bins,LRGscat[:,0],LRGscat[:,1],np.float32(z_redshift),periodic=True, verbose=True)
        SHAM_array = wp_dat['wp']
    return SHAM_array
    

# chi2
def chi2(sigma_M,sigma_V,M_ceil):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(float32(sigma_M)),repeat(float32(sigma_V)),repeat(float32(M_ceil))))
    if function == 'mps':
         # average the result for multiple seeds
        xi0,xi2,xi4 = mean(xi0_tmp,axis=0,dtype='float32')[0],mean(xi0_tmp,axis=0,dtype='float32')[1],mean(xi0_tmp,axis=0,dtype='float32')[2]
        # identify the fitting multipoles
        if multipole=='mono':
            model = xi0
        elif multipole=='quad':
            model = append(xi0,xi2)
        else:
            model = append(xi0,xi2,xi4)
    else:
        model = mean(xi0_tmp,axis=0,dtype='float32')
        mocks = mocks[binmin:binmax]
    # calculate the covariance, residuals and chi2
    Nbins = len(model)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    return res.dot(covR.dot(res))

# prior
def prior(cube, ndim, nparams):
    global prior_min,prior_max
    if gal=='LRG':
        if GC == 'NGC':
            cube[0] = 1.0*cube[0]+0.7
            cube[1] = 50*cube[1]+80  
            cube[2] = 2000*cube[2]+1000 
            prior_min = [0.7,80,1000]
            prior_max = [1.7,130,3000]

        else:
            cube[0] = 1.5*cube[0]+0.5 
            cube[1] = 70*cube[1]+70  
            cube[2] = 1600*cube[2]+600  
            prior_min = [0.5,70,600]
            prior_max = [2.0,140,2200]
    else:
        if GC == 'NGC':
            cube[0] = 1.4*cube[0]+0.6    
            cube[1] = 65*cube[1]      
            cube[2] = 600*cube[2]+200             
            prior_min = [0.6,0,200]
            prior_max = [2.0,65,800]
        else:
            cube[0] = 1.4*cube[0]+0.4   
            cube[1] = 60*cube[1]      
            cube[2] = 600*cube[2]+200             
            prior_min = [0.4,0,200]
            prior_max = [1.8,60,800]


# loglikelihood = -0.5*chi2    
def loglike(cube, ndim, nparams):
    sigma_high,sigma,vhigh = cube[0],cube[1],cube[2]
    return -0.5*chi2(sigma_high,sigma,vhigh)   

# number of dimensions our problem has
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)

# run MultiNest & write the parameter's name
pymultinest.run(loglike, prior, npar,n_live_points= npoints, outputfiles_basename=fileroot, resume =True, verbose = True,n_iter_before_update=5,write_output=True)
f=open(fileroot+'.paramnames', 'w')
for param in parameters:
    f.write(param+'\n')
f.close()

# prior ranges
f=open(fileroot+'.ranges', 'w')
for i,param in enumerate(parameters):
    f.write('{}  {} {}\n'.format(param,prior_min[i],prior_max[i]))
f.close()

# getdist plot
sample = loadMCSamples(fileroot)
plt.rcParams['text.usetex'] = False
g = plots.get_single_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
#g.settings.title_limit_fontsize = 14
g = plots.get_subplot_plotter()
g.triangle_plot(sample,parameters, filled=True)#,title_limit=1)
g.export('posterior_{}_{}_{}_{}_{}_z{}-{}.pdf'.format(date,gal,GC,function,rscale,zmin,zmax))
plt.close('all')
# results
print('Results:')
stats = sample.getMargeStats()
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
sigma = np.zeros(npar)
for i in range(npar):
    par = stats.parWithName(parameters[i])
    #best[i] = par.bestfit_sample
    mean[i] = par.mean
    sigma[i] = par.err
    lower[i] = par.limits[0].lower
    upper[i] = par.limits[0].upper
    best[i] = (lower[i] + upper[i]) * 0.5
    print('getdist {0:s}: [{1:.6f}, {2:.6f}]'.format( \
        parameters[i],  lower[i], upper[i]))


a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
stats = a.get_stats()    
fin = time.time()  
    
file = gal+'_'+GC+'_Vzsmear_report.txt'
f = open(file,'a')
f.write('the total {} in {} SHAM costs {:.6} s in 16 cores \n'.format(gal,GC,fin-init))
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
