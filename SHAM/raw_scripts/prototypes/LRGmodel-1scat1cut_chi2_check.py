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
halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 
halo = fits.open(halofile)

if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
    
halo.close()
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac[:,i] = np.copy(data[key])

half = int(len(datac)/2)
## find the best parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

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
    datav = np.copy(data[var])
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
    if nseed<32:
        with Pool(processes = nseed) as p:
            xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(Sigma)))
    else:
        with Pool(processes = 30) as p:
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


# generate random number arrays once and for all

for nseed in [30]:
    ini = time.time()
    uniform_randoms = [np.random.RandomState(seed=int(x+1)).rand(len(datac)) for x in range(nseed)]
    print('uniform random number arrays are ready.')

    fc = open('a_LRG_'+GC+'_'+multipole+'_robust_'+var+'_nseed'+str(nseed)+'.txt','a')
    if GC=='NGC':
        x =np.linspace(0.14,0.2,36) # Vmax
    if GC=='SGC':
        x =np.linspace(0.196,0.238,26) # Vmax
    #x = np.linspace(0.3,0.332,26)  # Vpeak
    for i in x:
        fc.write('{} {:.5}\n'.format(i,chi2(i)))       
    fin = time.time()
    fc.write('# nseed = {} with {} points costs {:.5} s'.format(nseed,len(x),fin-ini))
    fc.close() 
    
'''    
MDV = data['Vpeak'] 
MDM = data['Vmax'] 
fig,ax = plt.subplots() 
plt.scatter(MDM,MDV,c='r',alpha=0.4,s=5) 
plt.ylabel('Vpeak ($km/s$)') 
plt.xlabel('Vmax ($km/s$)') 
plt.plot(np.linspace(23,2100,10),np.linspace(23,2100,10),'k--',label='$Vpeak=Vmax$') 
plt.title('Vpeak - Vmax relation for subhaloes') 
plt.savefig('Vpeak - Vmax.png')
plt.close()



fig=plt.figure()
ax = plt.subplot2grid((1,2),(0,1))
#for sigma in [0.1,0.3]:
for sigma,vhigh in zip([0.1,0.1,0.3],(200,400,200)):
    uniform=uniform_randoms[0]
    datav = np.copy(data['Vmax'])   
    rand1 = np.append(sigma*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    #datav*=( 1+rand1)
    #LRGorg = datac[:,4][np.argpartition(-datav,LRGnum)[:LRGnum]]
    #n2,bins2=np.histogram(LRGorg,bins=50,range=(10,1000))
    #ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5,label = '$\sigma=$'+str(sigma))
    #plt.title('LRG {} distribution'.format(GC))
    datav*=( 1+rand1)
    org3  = datac[(datav<vhigh)]
    LRGorg = org3[:,4][np.argpartition(-datav[(datav<vhigh)],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(10,1000))
    
    ax.plot(bins[:-1],n2/n,alpha=0.5,label='$\sigma=$'+str(sigma)+',Vcut='+str(vhigh)+'km/s')
    plt.title('ELG {} distribution'.format(GC))
    plt.legend(loc=0)
    plt.ylabel('# of galaxies in 1 halo')
    plt.xlabel('Vmax (km/s)')
    ax.set_xlim(1000,10)
    
plt.savefig('Vmax_distr.png',bbox_tight=True)
plt.close()


'''