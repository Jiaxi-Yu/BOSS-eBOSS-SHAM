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
date     = '0810' 
npoints  = 100 
nseed    = 30
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
fileroot = 'MCMCout/3-param_'+date+'/'+gal+'_'+GC+'/multinest_'


if gal == 'LRG':
    LRGnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS+SEQUELS_'+gal+'_'+GC+'_v7_2.dat'
    halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 

if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
    halofile = home+'catalog/UNIT_hlist_0.53780.fits.gz'
    
# cosmological parameters
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

# generate covariance matrices and observations
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'  
#randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
# Read the covariance matrices and observations
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
    print('first begins')
    x00,x20,x40=sham_cal(uni,sigM,sigV,Mtrun)
    print('second begins')
    x01,x21,x41=sham_cal(uni1,sigM,sigV,Mtrun)
    return [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2]

def sham_cal(uniform,sigma_high,sigma,v_high):
    init = time.time()
    datav = datac[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    fin = time.time()  
    print('SHAM scattering costs {:.6} s \n'.format(fin-init))
    init = time.time()
    LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]
    #LRGscat = org3[argpartition(-datav[ind],LRGnum)[:(LRGnum)]] #3.06s
    fin = time.time()  
    print('SHAM selecting Nth largest costs {:.6} s \n'.format(fin-init))

    
    # transfer to the redshift space
    scathalf = int(len(LRGscat)/2)
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
    return [xi0_single,xi2_single,xi4_single]

# chi2
def chi2(sigma_M,sigma_V,M_ceil):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(float32(sigma_M)),repeat(float32(sigma_V)),repeat(float32(M_ceil))))
    
     # average the result for multiple seeds
    xi0,xi2,xi4 = mean(xi0_tmp,axis=0,dtype='float32')[0],mean(xi0_tmp,axis=0,dtype='float32')[1],mean(xi0_tmp,axis=0,dtype='float32')[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        mocks = hdu[1].data[multipole][binmin:binmax,:]
        covcut = cov(mocks).astype('float32')
        OBS   = obscf['col3'].astype('float32')
    if multipole=='quad':
        model = append(xi0,xi2)
        mocks = vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col3'],obscf['col4']).astype('float32')  
    if multipole=='hexa':
        model = append(xi0,xi2,xi4)
        mocks = vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:],hdu[1].data[multipole][binmin+400:binmax+400,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col3'],obscf['col4'],obscf['col5']).astype('float32')

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
            cube[0] = 1.0*cube[0]+0.4 # uniform between [0.4,1.4]
            cube[1] = 50*cube[1]+80  # uniform between [80,130]
            cube[2] = 1000*cube[2]+800  # uniform between [800,1800]
        if GC == 'SGC':
            cube[0] = 1.1*cube[0]+0.4 # uniform between [0.4,1.5]
            cube[1] = 50*cube[1]+80  # uniform between [80,130]
            cube[2] = 1400*cube[2]+600  # uniform between [600,2000]
    
    if gal=='ELG':
        cube[0] = 1.5*cube[0]    # uniform between [0,1.5]
        cube[1] = 100*cube[1]      # uniform between [0,100]
        cube[2] = 550*cube[2]+50     # uniform between [50,600]
    prior_min = [cube[0][0],cube[1][0],cube[2][0]]
    prior_max = [cube[0][1],cube[1][1],cube[2][1]]


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
prior_min = [x*0 for x in range(npar)]
prior_max = [x*0 for x in range(npar)]
prior([np.array([0,1]),np.array([0,1]),np.array([0,1])],None,None)
f=open(fileroot+'.ranges', 'w')
for i,param in enumerate(parameters):
    f.write('{}  {} {}\n'.format(param,prior_min[i],prior_max[i]))
f.close()

fin = time.time()  
print('the total {} in {} SHAM costs {:.6} s \n'.format(gal,GC,fin-init))

# getdist plot
sample = loadMCSamples(fileroot)
plt.rcParams['text.usetex'] = False
g = plots.get_single_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
#g.settings.title_limit_fontsize = 14
g = plots.get_subplot_plotter()
g.triangle_plot(sample,parameters, filled=True)#,title_limit=1)
g.export('{}_{}_{}_posterior.pdf'.format(date,gal,GC))
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
    
file = gal+'_'+GC+'_Vzsmear_report.txt'
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
