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
gal      = 'LRG'
GC       = sys.argv[1]
date     = '0526' #'0523' '0523-1' '0526'
func = 'HAM'
npoints  = 200#int(sys.argv[3])
nseed    = 5
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
fileroot = 'MCMCout/'+date+'/'+func+'_'+gal+'_'+GC+'/multinest_'


if gal == 'LRG':
    LRGnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    mockdir  = '/global/cscratch1/sd/zhaoc/EZmock/2PCF/LRGv7_syst/z'+str(zmin)+'z'+str(zmax)+'/2PCF/'
    obsname  = home+'catalog/pair_counts_s-mu_pip_eBOSS+SEQUELS_'+gal+'_'+GC+'_v7_2.dat'
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
datac = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac[:,i] = np.copy(data[key])
#V = np.copy(data[var]).astype('float32')
datac = datac.astype('float32')
half = int(len(data)/2)

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(data)).astype('float32') for x in range(nseed)] 

# HAM application
def sham_tpcf1(uniform,sigma_high,N1,v_high):
    datav = np.copy(datac[:,-1])
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    ind_low   = np.argpartition(-datav,int(N1))[int(N1):]  # scattering low
    org3  = datac[ind_low][datav[ind_low]<v_high]  # 4.89s low end cut the intermediate
    LRGscat = np.vstack((datac[np.argpartition(-datav,int(N1))[:int(N1)]],org3[np.argpartition(-datav[ind_low][datav[ind_low]<v_high],LRGnum-int(N1))[:(LRGnum-int(N1))]])) #3.06s
    n1,bins1=np.histogram(datac[np.argpartition(-datav,int(N1))[:int(N1)]][:,-1],bins=50,range=(0,1500))
    n2,bins2=np.histogram(org3[np.argpartition(-datav[ind_low][datav[ind_low]<v_high],LRGnum-int(N1))[:(LRGnum-int(N1))]][:,-1],bins=50,range=(0,1500))
    n3,bins3=np.histogram(LRGscat[:,-1],bins=50,range=(0,1500))
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
    return [xi0_single,xi2_single,xi4_single,n1,n2,n3]

def sham_tpcf(uniform,sigma_high,v_high):
    datav = np.copy(datac[:,-1])
    # shuffle the halo catalogue and select those have a galaxy inside
    rand1 = np.append(sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav*=( 1+rand1) #0.5s
    org3  = datac[(datav<v_high)]  # 4.89s
    LRGscat = org3[np.argpartition(-datav[(datav<v_high)],LRGnum)[:(LRGnum)]] #3.06s
    n2,bins2=np.histogram(LRGscat[:,-1],bins=50,range=(0,1500))
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
    return [xi0_single,xi2_single,xi4_single,n2]


# plot the best-fit
with Pool(processes = nseed) as p:
    if GC=='NGC':
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(0.8003797650857304)),repeat(np.float32(1167.7109330976073)))) 
        xi1_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(0)),repeat(int(1e4)),repeat(np.float32(330.7109330976073))))
        #xi2_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(0.8003797650857304)),repeat(int(1e3)),repeat(np.float32(1167.7109330976073)))) 
    if GC=='SGC':
        xi_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(0.7104378528883745)),repeat(np.float32(993.8109564008059)))) 
        xi1_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(0.7104378528883745)),repeat(int(5e3)),repeat(np.float32(793.8109564008059))))    
        #xi2_ELG = p.starmap(sham_tpcf1,zip(uniform_randoms,repeat(np.float32(0.7104378528883745)),repeat(int(1e3)),repeat(np.float32(993.8109564008059))))

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

if multipole=='quad':
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(['col3','col4'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscf[col]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none")
            ax[j,k].plot(s,s**2*(np.mean(xi_ELG,axis=0)[k]-values[j]),c='m',alpha=0.6)
            ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
            #ax[j,k].plot(s,s**2*(np.mean(xi2_ELG,axis=0)[k]-values[j]),c='b',alpha=0.9)
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                label = ['Multinest','bimodel N1=1e4','obs']#['Multinest','bimodel N1=1e4','bimodel N1=1e3','obs']
                plt.legend(label,loc=1)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

    plt.savefig('bimodel_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()

    
#nlist = [xi_ELG[x][3] for x in range(nseed)]
#narray = np.array(nlist).T

n1list = [xi1_ELG[x][3] for x in range(nseed)]
n1array = np.array(n1list).T
n11list = [xi1_ELG[x][4] for x in range(nseed)]
n11array = np.array(n11list).T
n12list = [xi1_ELG[x][5] for x in range(nseed)]
n12array = np.array(n12list).T

#n2list = [xi2_ELG[x][3] for x in range(nseed)]
#n2array = np.array(n1list).T
#n21list = [xi2_ELG[x][4] for x in range(nseed)]
#n21array = np.array(n21list).T

# plot the galaxy probability distribution and the real galaxy number distribution 
n,bins=np.histogram(datac[:,-1],bins=50,range=(0,1500))
fig =plt.figure(figsize=(16,6))
ax = plt.subplot2grid((1,2),(0,0))
binmid = (bins[:-1]+bins[1:])/2
#ax.errorbar(binmid,np.mean(xi_ELG,axis=0)[3]/(n),yerr = np.std(narray,axis=-1)/(n),color='orange',alpha=0.7,ecolor='orange',label='tick model',ds='steps-mid')
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[3]/(n),yerr=np.std(n1array,axis=-1)/(n),color='c',alpha=0.7,ecolor='c',label='bimodel-high',ds='steps-mid')
plt.ylabel('prob. to have 1 galaxy in 1 halo')
plt.legend(loc=2)
plt.title('Vpeak probability distribution: {} in {} '.format(gal,GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1510,10)
ax = plt.subplot2grid((1,2),(0,1))
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[4]/(n),yerr=np.std(n11array,axis=-1)/(n),color='b',alpha=0.7,ecolor='b',label='bimodel-low',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[5]/(n),yerr=np.std(n12array,axis=-1)/(n),color='orange',alpha=0.7,ecolor='orange',label='bimodel',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi2_ELG,axis=0)[3]/(n),yerr=np.std(n2array,axis=-1)/(n),color='r',alpha=0.7,ecolor='r',label='bimodel-high',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi2_ELG,axis=0)[4]/(n),yerr=np.std(n21array,axis=-1)/(n),color='m',alpha=0.7,ecolor='m',label='bimodel-low',ds='steps-mid')
plt.ylabel('prob. to have 1 galaxy in 1 halo')
plt.legend(loc=2)
plt.title('Vpeak probability distribution: {} in {} '.format(gal,GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1510,10)
'''
ax = plt.subplot2grid((1,2),(0,1))
#ax.errorbar(binmid,np.mean(xi_ELG,axis=0)[3],yerr = np.std(narray,axis=-1),color='orange',alpha=0.7,ecolor='orange',label='tick model',ds='steps-mid')
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[3],yerr=np.std(n1array,axis=-1),color='c',alpha=0.7,ecolor='c',label='bimodel-high',ds='steps-mid')
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[4],yerr=np.std(n11array,axis=-1),color='b',alpha=0.7,ecolor='b',label='bimodel-low',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[5],yerr=np.std(n12array,axis=-1),color='orange',alpha=0.7,ecolor='orange',label='bimodel',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi2_ELG,axis=0)[3],yerr=np.std(n2array,axis=-1),color='r',alpha=0.7,ecolor='r',label='bimodel-high',ds='steps-mid')
#ax.errorbar(binmid,np.mean(xi2_ELG,axis=0)[4],yerr=np.std(n21array,axis=-1),color='m',alpha=0.7,ecolor='m',label='bimodel-low',ds='steps-mid')
ax.step(binmid,n,color='k',label='UNIT sim.')
plt.yscale('log')
plt.ylabel('galaxy numbers')
plt.legend(loc=2)
plt.title('Vpeak distribution: {} in {}'.format(gal,GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1510,10)
'''
plt.savefig('MCMC_distr_'+gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
