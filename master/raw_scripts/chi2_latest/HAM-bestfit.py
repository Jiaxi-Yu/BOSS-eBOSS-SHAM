import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
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
from iminuit import Minuit
import matplotlib.gridspec as gridspec 
import sys

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
nseed    = 15
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

# covariance matrix and the observation 2pcf path
if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    precut   = 80
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/ELGv7_nosys_rmu/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS_'+gal+'_'+GC+'_v7.dat'
    halofile = home+'catalog/UNIT_hlist_0.53780.fits.gz' 
if gal == 'LRG':
    LRGnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    precut   = 160
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/eBOSS_'+gal+'_clustering_'+GC+'_v7_2.dat.fits'
    halofile = home+'catalog/UNIT_hlist_0.58760.fits.gz' 

# cosmological parameters
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate separation bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
if rscale=='log':
    nbins = 50
    bins=np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
    print('Note: the covariance matrix should also change to log scale.')
if (rmax-rmin)/nbins!=1:
    warnings.warn("the fitting should have 1Mpc/h bin to match the covariance matrices and observation.")
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

print('selecting only the necessary variables...')
datac = np.zeros((len(data),4))
for i,key in enumerate(['X','Y','Z','VZ']):
    datac[:,i] = np.copy(data[key])
V = np.copy(data[var]).astype('float32')
datac = datac.astype('float32')
half = int(len(data)/2)

# generate nseed Gaussian random number arrays in a list
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(data)).astype('float32') for x in range(nseed)] 
print('the uniform random number dtype is ',uniform_randoms[0].dtype)

# generate covariance matrices and observations
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'  
randname = obsname[:-8]+'ran.fits'
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
covmatrix(home,mockdir,covfits,gal,GC,zmin,zmax,Om,os.path.exists(covfits))
obs(home,gal,GC,obsname,randname,obs2pcf,rmin,rmax,nbins,zmin,zmax,Om,os.path.exists(obs2pcf))
# Read the covariance matrices and observations
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')


# HAM application
def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:LRGnum]] #3.06s
    # transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+LRGscat[:,3]*(1+z)/H)
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

#from functools import partial
def chi2(sigma_high,v_high):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(sigma_high)),repeat(np.float(v_high))))
    print('a second calculation')
    with Pool(processes = nseed) as p:
        xi1_tmp = p.starmap(sham_tpcf,zip(uniform_randoms1,repeat(np.float32(sigma_high)),repeat(np.float(v_high))))
    
     # average the result for multiple seeds
    xi0,xi2,xi4 = (np.mean(xi0_tmp,axis=0,dtype='float32')[0]+np.mean(xi1_tmp,axis=0,dtype='float32')[0])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[1]+np.mean(xi1_tmp,axis=0,dtype='float32')[1])/2,(np.mean(xi0_tmp,axis=0,dtype='float32')[2]+np.mean(xi1_tmp,axis=0,dtype='float32')[2])/2

    # identify the fitting multipoles
    if multipole=='mono':
        model = xi0
        mocks = hdu[1].data[multipole][binmin:binmax,:]
        covcut = np.cov(mocks).astype('float32')
        OBS   = obscf['col3'].astype('float32')
    if multipole=='quad':
        model = np.append(xi0,xi2)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:]))
        covcut  = np.cov(mocks).astype('float32')
        OBS   = np.append(obscf['col3'],obscf['col4']).astype('float32')  
    if multipole=='hexa':
        model = np.append(xi0,xi2,xi4)
        mocks = np.vstack((hdu[1].data[multipole][binmin:binmax,:],hdu[1].data[multipole][binmin+200:binmax+200,:],hdu[1].data[multipole][binmin+400:binmax+400,:]))
        covcut  = np.cov(mocks).astype('float32')
        OBS   = np.append(obscf['col3'],obscf['col4'],obscf['col5']).astype('float32')

    # calculate the covariance, residuals and chi2
    Nbins = len(model)
    covR  = np.linalg.inv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
    res = OBS-model
    fc.write('{} {} {} \n'.format(sigma_high,v_high,res.dot(covR.dot(res))))
    return res.dot(covR.dot(res))

# record parameter sets and chi2
chifile1 = 'HAM-bestfit_'+gal+'_'+GC+'_param-chi2.txt' 
fc=open(chifile1,'a')    
fc.write('# sigma_high  v_high  chi2 \n')
# run optimiser
time_start=time.time()
sigma = Minuit(chi2,sigma_high=0.3,v_high=300.0,limit_sigma_high=(0,1),limit_v_high=(100,1000),error_sigma_high=0.03,error_v_high=30,errordef=0.5) 
sigma.migrad(precision=0.001)
fc.close()

# report final result conclusion
chifile = 'HAM-bestfit_'+gal+'_'+GC+'_report.txt'  
f=open(chifile,'a')
f.write(gal+' '+GC+': \n')
f.write(str(sigma.get_fmin())+'\n')
f.write(str(sigma.values)+'\n')
f.write(str(sigma.errors)+'\n')
f.write('chi2 : '+str(sigma.fval)+'\n')
f.write('reduced chi2 should be around 1 : {:.5} \n'.format(sigma.fval / (len(s) - 2)))
f.write(str(sigma.get_param_states())+'\n')
f.write('{} in {}:sigma_high, v_high = {:.3},{:.6} km/s \n'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
time_end=time.time()
f.write('chi-square fitting finished, costing {:.5} s \n'.format(time_end-time_start))
f.close()
'''
# plot the galaxy probability distribution and the real galaxy number distribution 
# plot the best-fit
with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(sigma.values['sigma_high'])),repeat(np.float32(sigma.values['v_high']))))


values = [np.zeros(nbins),np.mean(xi_ELG,axis=0)[j]]    
if multipole=='mono':    
    fig = plt.figure(figsize = (14, 8))
spec = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[4, 1], hspace=0.3, wspace=0.4)
ax = np.empty((2,2), dtype=type(plt.axes))
for j in range(2):
    for k,pole in zip(range(2),['mono','quad']):
        ax[k,j] = fig.add_subplot(spec[k,j])
        value =a[k]
        for i,xi in enumerate(xi_LRG):
            ax[k,j].plot(s,s**2*(xi[j]-value),lw=0.3,alpha=0.6)
        
        ax[k,j].plot(s,s**2*(np.mean(xi_LRG,axis=0)[j]-value),c='k',label='mean')
        ax[k,j].errorbar(s,s**2*obscf['col3'],s**2*errbar[binmin:binmax], marker='^',ecolor='k',ls="none")
        errorbar(s,s**2*(np.mean(xi_LRG,axis=0)[j]-value),s**2*np.std(xi_LRG,axis=0)[j],ecolor='k',ls="none")
        if (k==0)&(j==0):
            #ax[k,j].set_ylim(75,105)
            ax[k,j].set_ylabel('$s^2*\\xi_0$') 
        if (k==0)&(j==1):
            #ax[k,j].set_ylim(-80,0)
            ax[k,j].set_ylabel('$s^2*\\xi_2$')     
        if (k==1)&(j==0):
            #ax[k,j].set_ylim(-2,2)
            ax[k,j].set_ylabel('$s^2[\\xi_0-mean(\\xi_0)]$')
        if (k==1)&(j==1):
            #ax[k,j].set_ylim(-5,5)
            ax[k,j].set_ylabel('$s^2[\\xi_2-mean(\\xi_2)]$')
        plt.legend(loc=0)
        plt.xlabel('s (Mpc $h^{-1}$)')
    plt.title('LRG {} in {} using {}'.format(pole,GC,var))
    fig,ax =plt.subplots(figsize=(8,6))
    ax.
    ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[0],c='m',alpha=0.5)
    label = ['best fit','obs']
    plt.legend(label,loc=0)
    plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
    plt.xlabel('s (Mpc $h^{-1}$)')
    plt.ylabel('s^2 * $\\xi_0$')
    plt.savefig('HAM-bestfit_cf_mono_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole=='quad':
    fig =plt.figure(figsize=(16,6))
    for col,covbin,k in zip(['col3','col4'],[int(0),int(200)],range(2)):
        ax = plt.subplot2grid((1,2),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('HAM-bestfit_cf_quad_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
if multipole == 'hexa':
    fig =plt.figure(figsize=(24,6))
    for col,covbin,k in zip(['col3','col4','col5'],[int(0),int(200),int(400)],range(3)):
        ax = plt.subplot2grid((1,3),(0,k)) 
        ax.errorbar(s,s**2*obscf[col],s**2*errbar[binmin+covbin:binmax+covbin], marker='^',ecolor='k',ls="none")
        ax.plot(s,s**2*np.mean(xi_ELG,axis=0)[k],c='m',alpha=0.5)
        label = ['best fit','obs']
        plt.legend(label,loc=0)
        plt.title('{} in {}: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
        plt.xlabel('s (Mpc $h^{-1}$)')
        plt.ylabel('s^2 * $\\xi_{}$'.format(k*2))
    plt.savefig('HAM-bestfit_cf_hexa_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()

# plot the galaxy probability distribution and the real galaxy number distribution 
n,bins=np.histogram(V,bins=50,range=(0,1000))
fig =plt.figure(figsize=(16,6))
ax = plt.subplot2grid((1,2),(0,0))
for uniform in uniform_randoms:
    datav = np.copy(V)   
    rand1 = np.append(sigma.values['sigma_high']*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma.values['sigma_high']*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = V[(datav<sigma.values['v_high'])]
    LRGorg = org3[np.argpartition(-datav[(datav<sigma.values['v_high'])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(0,1000))
    ax.plot(bins[:-1],n2/n,alpha=0.5,lw=0.5)
plt.ylabel('prob. to have 1 galaxy in 1 halo')
plt.title('{} {} distribution: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)
    
ax = plt.subplot2grid((1,2),(0,1))
for uniform in uniform_randoms:
    datav = np.copy(V)   
    rand1 = np.append(sigma.values['sigma_high']*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma.values['sigma_high']*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) 
    datav*=( 1+rand1)
    org3  = V[(datav<sigma.values['v_high'])]
    LRGorg = org3[np.argpartition(-datav[(datav<sigma.values['v_high'])],LRGnum)[:LRGnum]]
    n2,bins2=np.histogram(LRGorg,bins=50,range=(0,1000))    
    ax.plot(bins[:-1],n2,alpha=0.5,lw=0.5)
ax.plot(bins[:-1],n,alpha=0.5,lw=0.5)
plt.yscale('log')
plt.ylabel('galaxy numbers')
plt.title('{} {} distribution: sigmahigh={:.3}, vhigh={:.6} km/s'.format(gal,GC,sigma.values['sigma_high'],sigma.values['v_high']))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)

plt.savefig('HAM-bestfit_distr_'+gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
'''

f=open(chifile,'a')
fin = time.time()  
f.write('the total {} in {} SHAM costs {:.6} s \n'.format(gal,GC,fin-init))
f.close()


