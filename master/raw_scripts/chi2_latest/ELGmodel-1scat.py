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
gal      = 'ELG' 
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 30
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
LRGnum   = int(4e5)
autocorr = 1
nseed    = 15

home      = '/global/cscratch1/sd/jiaxi/master/'
zmin     = 0.6
zmax     = 1.1
z = 0.8594

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
halofile = home+'catalog/UNIT_hlist_0.53780-cut.fits.gz' 
halo = fits.open(halofile)
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
    
halo.close()
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ','Vpeak']):
    datac[:,i] = np.copy(data[key])

half = int(len(data)/2)
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate random number arrays once and for all
uniform_randoms = [np.random.RandomState(seed=int(x+1)).rand(len(data)) for x in range(nseed)]
print('uniform random number arrays are ready.')

#for GC in ['NGC','SGC']: 
GC = sys.argv[1]
#GC='NGC'
mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'   #cov_'+GC+'_->corrcoef
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'

covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))

# Read the covariance matrix and 
hdu = fits.open(covfits) # cov([mono,quadru])
cov = hdu[1].data['cov'+multipole]
Nbins,Nmock = (hdu[1].data[multipole]).shape # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(data['Vpeak'])
    
    # shuffle the halo catalogue and select those have a galaxy inside    
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = datac[(datav<v_high)]
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:LRGnum]]

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
def chi2(sigma_high,v_high):
# calculate mean monopole in parallel
    print('parallel calculation')
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(sigma_high),repeat(v_high)))
    
    # average the result for multiple seeds
    xi0,xi2,xi4 = np.mean(xi0_tmp,axis=0)[0],np.mean(xi0_tmp,axis=0)[1],np.mean(xi0_tmp,axis=0)[2]

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        covcut = cov[binmin:binmax,binmin:binmax]
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = obscf['col3']
    elif multipole=='quad':
        model = np.append(xi0,xi2)
        covlen = int(len(cov)/2)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+covlen:binmax+covlen,:]))
        covcut  = np.hstack((cov_tmp[:,binmin:binmax],cov_tmp[:,binmin+covlen:binmax+covlen]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col3'],obscf['col4'])  # obs([mono,quadru])
    else:
        model = np.append(xi0,xi2,xi4)
        covlen = int(len(cov)/3)
        cov_tmp = np.vstack((cov[binmin:binmax,:],cov[binmin+covlen:binmax+covlen,:],cov[binmin+2*covlen:binmax+2*covlen,:]))
        covcut  = np.hstack((cov_tmp[:,binmin:binmax],cov_tmp[:,binmin+covlen:binmax+covlen],cov_tmp[:,binmin+2*covlen:binmax+2*covlen]))
        covR  = np.linalg.inv(covcut)*(Nmock-(Nbins-binmin)-2)/(Nmock-1)
        OBS   = np.append(obscf['col3'],obscf['col4'],obscf['col5'])

        ### calculate the covariance, residuals and chi2
    res = OBS-model
    fc.write('{} {} {} \n'.format(sigma_high,v_high,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))

 
# chi2 minimise
chifile = gal+'_results_1scat.txt'
f=open(chifile,'a')
f.write(gal+' '+GC+': \n')
chifile1 = gal+'_param+chi2_1scat.txt'
fc=open(chifile1,'a')
fc.write('# sigma_high  v_high  chi2 \n')
time_start=time.time()
print('chi-square fitting starts...')
## method 1：Minute-> failed because it seems to be lost 
sigma = Minuit(chi2,sigma_high=0.3,v_high=100.0,limit_sigma_high=(0,2),limit_v_high=(100,500),errordef=1) 
sigma.migrad(precision=0.01)  # run optimiser
time_end=time.time()
f.write('parallel calculation best param sigma_high, v_high = {:.3},{:.6} km/s \n'.format(sigma.values[0],sigma.values[1]))
#
f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
fc.close()

# plot the best fit result
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(sigma.values[0]),repeat(sigma.values[1])))

    
if multipole=='mono':    
    fig,ax =plt.subplots(figsize=(8,6))
    ax.errorbar(s,s**2*obscf['col3'],s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
    ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[0],c='m',alpha=0.5)
    label = ['best fit','obs']
    plt.legend(label,loc=0)
    plt.title('ELG in {}: sigmahigh={:.4}, vhigh={:.6} km/s'.format(GC,sigma.values[0],sigma.values[1]))
    plt.xlabel('d_cov (Mpc $h^{-1}$)')
    plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_mono_bestfit_'+gal+'_'+GC+'_1scat.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig =plt.figure(figsize=(16,6))
    for col,covbin,k in zip(['col3','col4'],[int(0),int(200)],range(2)):
        ax = plt.subplot2grid((1,2),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('ELG in {}: sigmahigh={:.4}, vhigh={:.6} km/s'.format(GC,sigma.values[0],sigma.values[1]))
        plt.xlabel('d_cov (Mpc $h^{-1}$)')
        plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_quad_bestfit_'+gal+'_'+GC+'_1scat.png',bbox_tight=True)
    plt.close()
if multipole == 'hexa':
    fig =plt.figure(figsize=(24,6))
    for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
        ax = plt.subplot2grid((1,3),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('ELG in {}: sigmahigh={:.4}, vhigh={:.6} km/s'.format(GC,sigma.values[0],sigma.values[1]))
        plt.xlabel('d_cov (Mpc $h^{-1}$)')
        plt.ylabel('d_cov^2 * $\\xi$')
    plt.savefig('cf_hexa_bestfit_'+gal+'_'+GC+'_1scat.png',bbox_tight=True)
    plt.close()

# also plot the galaxy probability distribution 
datav = np.copy(data['Vpeak']) 
n,bins=np.histogram(datav,bins=50,range=(10,1000))
fig,ax=plt.subplots()
for uniform in uniform_randoms:
    datav = np.copy(data['Vpeak'])   
    rand1 = np.append(sigma.values[0]*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma.values[0]*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = datac[(datav<sigma.values[1])]
    LRGorg = org3[:,4][np.argpartition(-datav[(datav<sigma.values[1])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(10,1000))
    
    ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5)
plt.title('ELG {} distribution: sigmahigh={:.4}, vhigh={:.6} km/s'.format(GC,sigma.values[0],sigma.values[1]))
plt.ylabel('# of galaxies in 1 halo')
plt.xlabel('Vmax (km/s)')
ax.set_xlim(1000,10)
plt.savefig('distributions'+gal+'_'+GC+'_1scat.png',bbox_tight=True)
plt.close()


fin = time.time()  
f.write('the total ELG SHAM costs {:.6} s \n'.format(fin-init))
f.close()

## LRG observation consistency