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
from obsbins import obsz
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
GC       = sys.argv[1]
sig = sys.argv[2]
Vceil = sys.argv[3]
func = 'HAM'
npoints  = 100#int(sys.argv[3])
nseed    = 10
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

if gal == 'ELG':
    LRGnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    precut   = 80
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

# no PIP
# generate covariance matrices and observations
covfits = home+'2PCF/obs/cov_'+gal+'_'+GC+'_'+multipole+'.fits.gz'  
obs2pcf  = home+'2PCF/obs/'+gal+'_'+GC+'.dat'
# Read the covariance matrices and observations
hdu = fits.open(covfits) # cov([mono,quadru])
Nmock = (hdu[1].data[multipole]).shape[1] # Nbins=np.array([Nbins,Nm])
errbar = np.std(hdu[1].data[multipole],axis=1)
hdu.close()
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')

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

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(data)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf1(uniform,sigma_high,sigma,v_high):
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:(LRGnum)]] #3.06s
    scathalf = int(len(LRGscat)/2)
    rand2 = np.append(sigma*np.sqrt(-2*np.log(uniform[:scathalf]))*np.cos(2*np.pi*uniform[-scathalf:]),sigma*np.sqrt(-2*np.log(uniform[:scathalf]))*np.sin(2*np.pi*uniform[-scathalf:]))
    # transfer to the redshift space
    z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+rand2)*(1+z)/H)
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

def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(V)
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:(LRGnum)]] #3.06s
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

#chi2 functions
def chi21(sigma_high,sigma,vhigh):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(sigma_high)),repeat(np.float32(sigma)),repeat(np.float32(vhigh))))
    with Pool(processes = nseed) as p:
        xi1_tmp = p.starmap(sham_tpcf1,zip(uniform_randoms1,repeat(np.float32(sigma_high)),repeat(np.float32(sigma)),repeat(np.float32(vhigh))))
    
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
    print('one chi2 completed.')
    return res.dot(covR.dot(res))

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
    return res.dot(covR.dot(res))



# plot the best-fit
with Pool(processes = nseed) as p:
    if GC=='NGC':
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(0.5129990708252805)),repeat(np.float32(268.1720066670134))))
        xi1_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(sig)),repeat(np.float32(21.9)),repeat(np.float32(Vceil)))) 
        a=chi21(sig,21.9,Vceil)
        print('Multinest +Vz smearing chi2: {:.7}'.format(a))
        print('Multinest chi2:',chi2(0.5129990708252805, 268.1720066670134))
    if GC=='SGC':
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(0.7896600001087524)),repeat(np.float32(342.6593190990616)))) 
        xi1_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(sig)),repeat(np.float(21.9)),repeat(np.float32(Vceil))))
        a =chi21(sig,21.9,Vceil) 
        print('Multinest +Vz smearing chi2: {:.7}',a)
        print('Multinest chi2:',chi2(0.7896600001087524, 342.6593190990616))

if multipole=='quad':
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(['col3','col4'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscf[col]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none")
            ax[j,k].plot(s,s**2*(np.mean(xi_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
            ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='r',alpha=0.5)
            #ax[j,k].plot(s,s**2*(np.mean(xi2_ELG,axis=0)[k]-values[j]),c='b',alpha=0.9)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                label = ['(sigma,Vceil)','(sigma,Vceil)+Vzsmear','obs']
                plt.legend(label,loc=0)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig('VZsmear_bothbest_{}_{}_chi2_{:.7}.png'.format(gal,GC,a),bbox_tight=True)
    plt.close()
    
#chi2 functions
'''
def chi21(sigma_high,sigma,vhigh):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(sigma_high)),repeat(np.float32(sigma)),repeat(np.float32(vhigh))))
    with Pool(processes = nseed) as p:
        xi1_tmp = p.starmap(sham_tpcf1,zip(uniform_randoms1,repeat(np.float32(sigma_high)),repeat(np.float32(sigma)),repeat(np.float32(vhigh))))
    
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
    print('one chi2 completed.')
    return res.dot(covR.dot(res))

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
    return res.dot(covR.dot(res))

'''