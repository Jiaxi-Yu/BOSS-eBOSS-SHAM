import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from astropy.table import Table
from astropy.io import ascii
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
from iminuit import Minuit
import sys


# variables
rscale = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
gal      = 'LRG' 
var      = 'Vmax'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = int(4e5)
autocorr = 1
nseed    = 30

home      = '/global/cscratch1/sd/jiaxi/master/'
zmin     = 0.6
zmax     = 1.0
z = 0.7018

# generate s and mu bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax

if rscale=='log':
    nbins = 50
    bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    print('Note: the covariance matrix should also change to log scale.')

s = (bins[:-1]+bins[1:])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

#covariance matrix and the observation 2pcf calculation
if (rmax-rmin)/nbins!=1:
    warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")

# RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=(RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf***************
print('reading the halo catalogue for creating the galaxy catalogue...')
ini = time.time()
halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 
halo = fits.open(halofile)

if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data    
halo.close()
'''
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac[:,i] = np.copy(data[key])
'''
datac = np.zeros((len(data),4))
for i,key in enumerate(['X','Y','Z','VZ']):
    datac[:,i] = np.copy(data[key])    
V = np.copy(data[var])

end = time.time()
print('{:.5} s'.format(end-ini))

half = int(len(datac)/2)
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate random number arrays once and for all
'''
uniform_randoms = [np.random.RandomState(seed=int(x+1)).rand(len(datac)) for x in range(nseed)]
'''
uniform_randoms = [np.random.RandomState(seed=int(time.time()*x)).randint(0,tmax,size=int(1e8),dtype='float32') for x in np.random.uniform(0,1,size=nseed)]

np.random.RandomState().randint(0, tmax, dtype='int') / np.float32(tmax)


uniform_randoms = [np.random.RandomState(seed=int(time.time()*x)).rand(len(datac)).astype('float32') for x in np.random.uniform(0,1,size=nseed)]
print('uniform random number arrays are ready.')


#for GC in ['NGC','SGC']:
GC = sys.argv[1]
#GC = 'NGC'
mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'


covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))

# Read the covariance matrix and 
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma):
    #datav = np.copy(data[var])
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    if gal=='LRG':
        ### shuffle and pick the Nth maximum values
        rand = np.append(sigma*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
        datav*=( 1+rand)
        LRGscat = datac[np.argpartition(-datav,LRGnum)[:LRGnum]]

    ### transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
    z_redshift %=boxsize
     ### count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
        ### calculate the 2pcf and the multipoles
    mono = DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    ### use trapz to integrate over mu
    xi0_single = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2_single = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4_single = np.trapz(hexa, dx=1./nmu, axis=-1)
    return [xi0_single,xi2_single,xi4_single]

#from functools import partial
def chi2(Sigma):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(Sigma)))
    
    # average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(xi0_tmp,axis=0)[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        mocks = hdu[1].data[multipole][binmin:binmax,:]
        covcut = np.cov(mocks)
        OBS   = obscf['col2']
    if multipole=='quad':
        model = np.append(xi0,xi2)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:]))
        covcut  = np.cov(mocks)
        OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
    if multipole=='hexa':
        model = np.append(xi0,xi2,xi4)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:],hdu[1].data[multipole][binmin+400:binmax+400,:]))
        covcut  = np.cov(mocks)
        OBS   = np.append(obscf['col2'],obscf['col3'],obscf['col4'])
    
    ### calculate the covariance, residuals and chi2
    Nbins = len(model)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    fc.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))


# chi2 minimise
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
if sys.argv[2]=='scipy':
    from scipy.optimize import minimize 
    chifile = gal+'_'+GC+'_results_scipy.txt'
    f=open(chifile,'a')
    f.write(gal+' '+GC+': \n')
    chifile1 = gal+'_'+GC+'_param+chi2_scipy.txt'
    fc=open(chifile1,'a')
    fc.write('# sigma chi2 \n')
    opt  = minimize(chi2,x0=0.3,bounds=[(0,1.0)])
    f.write(str(opt.message)+'\n')
    f.write(str(opt.nfev)+'\n')
    f.write(str(opt.success)+'\n')
    f.write(str(opt.x)+'\n')
else:
    chifile = gal+'_'+GC+'_'+var+'_results.txt'
    f=open(chifile,'a')
    f.write(gal+' '+GC+': \n')
    chifile1 = gal+'_'+GC+'_'+var+'_param+chi2.txt'
    fc=open(chifile1,'a')
    fc.write('# sigma chi2 \n')
    sigma = Minuit(chi2,Sigma=0.3,limit_Sigma=(0,0.7),error_Sigma=0.1,errordef=0.5)
    sigma.migrad(precision=0.001)  # run optimiser
    f.write(str(sigma.get_fmin())+'\n')
    f.write(str(sigma.values)+'\n')
    f.write(str(sigma.errors)+'\n')
    f.write('0.5*chi2 : '+str(sigma.fval)+'\n')
    f.write(str(sigma.get_param_states())+'\n')
    f.write('this value should be around 1: {:.5} \n'.format(sigma.fval / (len(s) - 2)))
    f.write('LRG in {}: sigma={:.4} \n'.format(GC,sigma.values['Sigma']))
    f.write('parallel calculation best param {:.3} \n'.format(sigma.values['Sigma']))
    
fc.close()
time_end=time.time()

f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))

fin = time.time()  
f.write('the total LRG SHAM costs {:.6} s \n'.format(fin-init))
f.close()


# plot the best fit result
with Pool(processes = nseed) as p:
    xi_LRG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(sigma.values['Sigma'])))
    
if multipole=='mono':    
    fig,ax =plt.subplots(figsize=(8,6))
    ax.errorbar(s,s**2*obscf['col2'],s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
    ax.plot(s,s**2*np.mean(xi_LRG,axis=0)[0],c='m',alpha=0.5)
    label = ['best fit','obs']
    plt.legend(label,loc=0)
    plt.title('LRG in {}: sigma={:.4} using {}'.format(GC,sigma.values['Sigma'],var))
    plt.xlabel('d_cov (Mpc $h^{-1}$)')
    plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_mono_bestfit_'+gal+'_'+GC+'_'+var+'.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig =plt.figure(figsize=(16,6))
    for col,covbin,k in zip(['col2','col3'],[int(0),int(200)],range(2)):
        ax = plt.subplot2grid((1,2),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_LRG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('LRG in {}: sigma={:.4} using {}'.format(GC,sigma.values['Sigma'],var))
        plt.xlabel('d_cov (Mpc $h^{-1}$)')
        plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_quad_bestfit_'+gal+'_'+GC+'_'+var+'.png',bbox_tight=True)
    plt.close()
if multipole=='hexa':
    fig =plt.figure(figsize=(24,6))
    for col,covbin,k in zip(['col2','col3','col4'],[int(0),int(200),int(400)],range(3)):
        ax = plt.subplot2grid((1,3),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_LRG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('LRG in {}: sigma={:.4} using {}'.format(GC,sigma.values['Sigma'],var))
        plt.xlabel('d_cov (Mpc $h^{-1}$)')
        plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_hexa_bestfit_'+gal+'_'+GC+'_'+var+'.png',bbox_tight=True)
    plt.close()

# also plot the galaxy probability distribution 
#datav = np.copy(data[var]) 
datav = np.copy(V)
n,bins=np.histogram(datav,bins=50,range=(10,1000))
fig,ax=plt.subplots()
for uniform in uniform_randoms:
    #datav = np.copy(data[var]) 
    datav=np.copy(V)
    rand1 = np.append(sigma.values['Sigma']*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma.values['Sigma']*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    #LRGorg = datac[:,4][np.argpartition(-datav,LRGnum)[:LRGnum]]
    LRGorg = V[np.argpartition(-datav,LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(10,1000))
    
    ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5)
plt.title('LRG {} distribution: Sigma={:.4} using {}'.format(GC,sigma.values['Sigma'],var))
plt.ylabel('# of galaxies in 1 halo')
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)
plt.savefig(gal+'_'+GC+'_1scat_'+var+'_distr.png',bbox_tight=True)
plt.close()

'''
# plot the mono+quad multipoles and present the scattering from 
'''


'''
# calculate mean multipoles in loop
#from functools import partial
xi0_tmp=[x for x in np.arange(nseed)]
def chi2_slow(Sigma):
    print('loop calculation')
    for i,uni in enumerate(uniform_randoms):
        xi0_tmp[i] = sham_tpcf(uni,Sigma)
    
    # average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(sigmsxi0_tmp,axis=0)[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        covcut = cov[binmin:binmax,binmin:binmax]
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = obscf['col2']
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        covlen = int(len(cov)/2)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+covlen:binmax+covlen,:]))
        covcut  = np.hstack((cov_tmp[:,binmin:binmax],cov_tmp[:,binmin+covlen:binmax+covlen]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col2'],obscf['col3'])  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)
        covlen = int(len(cov)/3)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+covlen:binmax+covlen,:],cov[binmin+2*covlen:binmax+2*covlen,:]))
        covcut  = np.hstack((cov_tmp[:,binmin:binmax],cov_tmp[:,binmin+covlen:binmax+covlen],cov_tmp[:,binmin+2*covlen:binmax+2*covlen]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col3'],obscf['col4'],obscf['col5'])
    
        ### calculate the covariance, residuals and chi2
    res = OBS-model
    fc.write('{} {} \n'.format(Sigma,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))
'''    


