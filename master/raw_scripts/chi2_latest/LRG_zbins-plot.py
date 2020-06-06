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
GC       = 'NGC'
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
RR=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
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
nfac = 133517.8 / 86486.0
# HAM application
def sham_tpcf(uniform,sigma_high,v_high,sigma_high1,v_high1):
    #NGC
    datav = np.copy(datac[:,-1])
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
    # SGC
    datav1 = np.copy(datac[:,-1])
    rand2 = np.append(sigma_high1*np.sqrt(-2*np.log(uniform[:half]))*np.cos(2*np.pi*uniform[half:]),sigma_high1*np.sqrt(-2*np.log(uniform[:half]))*np.sin(2*np.pi*uniform[half:])) # 2.9s
    datav1*=( 1+rand2) #0.5s
    org1  = datac[(datav1<v_high1)]  # 4.89s
    LRGscat1 = org1[np.argpartition(-datav1[(datav1<v_high1)],LRGnum)[:LRGnum]] #3.06s
    # transfer to the redshift space
    z_redshift1  = (LRGscat1[:,2]+LRGscat1[:,3]*(1+z)/H)
    z_redshift1 %=boxsize
    # count the galaxy pairs and normalise them
    DD_counts1 = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat1[:,0],LRGscat1[:,1],z_redshift1,periodic=True, verbose=True,boxsize=boxsize)
    DD = (DD_counts1['npairs']+DD_counts['npairs'] * nfac**2) / (1 + nfac**2)
    # calculate the 2pcf and the multipoles
    mono = (DD.reshape(nbins,nmu)/(LRGnum**2)/RR-1)
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    # use trapz to integrate over mu
    xi0_single = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2_single = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4_single = np.trapz(hexa, dx=1./nmu, axis=-1)
    return [xi0_single,xi2_single,xi4_single]



# plot the best-fit
with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(uniform_randoms,repeat(np.float32(0.8003797650857304)),repeat(np.float32(1167.7109330976073)),repeat(np.float32(0.7104378528883745)),repeat(np.float32(993.8109564008059))))    

'''
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

for zlow,zhigh in zip([0.65,0.6,0.75],[0.8,0.7,1.0]):
    # generate covariance matrices and observations
    covfits1 = home+'2PCF/zbins/'+gal+'_'+GC+'_z'+str(zlow)+'z'+str(zhigh)+'_'+multipole+'.fits.gz'  
    obs2pcf1  = home+'2PCF/zbins/'+gal+'_'+GC+'_z'+str(zlow)+'z'+str(zhigh)+'.dat'
    # Read the covariance matrices and observations
    hdu1 = fits.open(covfits1) # cov([mono,quadru])
    errbar1 = np.std(hdu1[1].data[multipole],axis=1)
    hdu1.close()
    obscf1 = Table.read(obs2pcf1,format='ascii.no_header')[binmin:binmax]  # obs 2pcf
    print('the covariance matrix and the observation 2pcf vector are ready.')
    if multipole=='quad':
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(['col3','col4'],[int(0),int(200)],['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),np.mean(xi1_ELG,axis=0)[k]]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none",alpha=0.5)
                ax[j,k].errorbar(s,s**2*(obscf1[col]-values[j]),s**2*errbar1[binmin+covbin:binmax+covbin],color='r', marker='o',ecolor='r',ls="none",alpha=0.5)
                ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]-values[j]),c='c',alpha=0.6)
                plt.xlabel('s (Mpc $h^{-1}$)')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))            
                    label = ['Multinest','obs 0.6<z<1.0','obs {}<z<{}'.format(zlow,zhigh)]
                    plt.legend(label,loc=1)
                    plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))

        plt.savefig(func+'-MCMC_cf_quad_bestfit_'+gal+'_'+GC+'_z'+str(zlow)+'.png',bbox_tight=True)
        plt.close()
'''        

# PIP NGC+SGC
obscf1 = Table.read(home+'catalog/pair_counts_s-mu_pip_eBOSS+SEQUELS_LRG_NGC_v7_2.dat',format='ascii.no_header')
obscf2 = Table.read(home+'catalog/pair_counts_s-mu_pip_eBOSS+SEQUELS_LRG_SGC_v7_2.dat',format='ascii.no_header')# obs 2pcf
print('the covariance matrix and the observation 2pcf vector are ready.')
nfac = 133517.8 / 86486.0
mu = (np.linspace(0,1,201)[1:]+np.linspace(0,1,201)[:-1])/2
dd = (obscf2['col3'] + obscf1['col3'] * nfac**2) / (1 + nfac**2)
dr = (obscf2['col4'] + obscf1['col4'] * nfac**2) / (1 + nfac**2)
rr = (obscf2['col5'] + obscf1['col5'] * nfac**2) / (1 + nfac**2)
mon = ((dd-2*dr+rr)/rr).reshape(250,200)
qua = mon * 2.5 * (3 * mu**2 - 1)
hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
## use trapz to integrate over mu
obs0 = np.trapz(mon, dx=1./200, axis=1)[binmin:binmax]
obs1 = np.trapz(qua, dx=1./200, axis=1)[binmin:binmax]
obs2 = np.trapz(hexad, dx=1./200, axis=1)[binmin:binmax]
 
hdu1 = fits.open(home+'2PCF/obs/cov_LRG_NGC_'+multipole+'.fits.gz') # cov([mono,quadru])Nbins=np.array([Nbins,Nm])
hdu2 = fits.open(home+'2PCF/obs/cov_LRG_SGC_'+multipole+'.fits.gz')
errbar = (np.std(hdu2[1].data[multipole],axis=1)+np.std(hdu1[1].data[multipole],axis=1)* nfac**2) / (1 + nfac)**2

hdu2.close()
hdu1.close()

if multipole=='quad':
    fig = plt.figure(figsize=(14,6))
    spec = gridspec.GridSpec(nrows=1,ncols=2,wspace=0.4)
    ax = np.empty((1,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip([obs0,obs1],[int(0),int(200)],['monopole','quadrupole'],range(2)):
        for j in range(1):
            ax[j,k] = fig.add_subplot(spec[j,k])
            plt.xlabel('s (Mpc $h^{-1}$)')
            plt.xlim(5,25)
            ax[j,k].errorbar(s,s**2*(col),s**2*errbar[binmin+covbin:binmax+covbin],color='k', marker='o',ecolor='k',ls="none",alpha=0.5,label = 'obs 0.6<z<1.0')
            ax[j,k].plot(s,s**2*(np.mean(xi1_ELG,axis=0)[k]),c='c',alpha=0.6,label='Multinest NGC+SGC')
            for zlow,zhigh in zip(['0.60','0.60','0.65','0.70','0.80'],['0.70','0.80','0.80','0.90','1.00']):
                # generate covariance matrices and observations
                obs2pcf  = home+'cheng_HOD_LRG/mps_log_LRG_NGC+SGC_eBOSS_v7_2_zs_'+zlow+'-'+zhigh+'.dat'
                obscf = Table.read(obs2pcf,format='ascii.no_header')
                ax[j,k].plot(obscf['col3'],(obscf['col'+str(k+4)]),alpha=0.6,label='obs {}<z<{}'.format(zlow,zhigh))
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))      
                plt.legend(loc=0)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if k==0:
                plt.ylim(50,110)
            if k==1:
                plt.ylim(-75,40)

    plt.savefig('zbins_'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()        


'''
nlist = [xi_ELG[x][3] for x in range(nseed)]
n1list = [xi1_ELG[x][3] for x in range(nseed)]
narray = np.array(nlist).T
n1array = np.array(n1list).T

# plot the galaxy probability distribution and the real galaxy number distribution 
n,bins=np.histogram(datac[:,-1],bins=50,range=(0,1000))
fig =plt.figure(figsize=(16,6))
ax = plt.subplot2grid((1,2),(0,0))
binmid = (bins[:-1]+bins[1:])/2
ax.errorbar(binmid,np.mean(xi_ELG,axis=0)[3]/(n+1),yerr = np.std(narray,axis=-1)/(n+1),color='m',alpha=0.7,ecolor='m',label='iminuit',ds='steps-mid')
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[3]/(n+1),yerr=np.std(n1array,axis=-1)/(n+1),color='c',alpha=0.7,ecolor='c',label='Multinest',ds='steps-mid')
plt.ylabel('prob. to have 1 galaxy in 1 halo')
plt.legend(loc=2)
plt.title('Vpeak probability distribution: {} in {} '.format(gal,GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)

ax = plt.subplot2grid((1,2),(0,1))
ax.errorbar(binmid,np.mean(xi_ELG,axis=0)[3],yerr = np.std(narray,axis=-1)[3],color='m',alpha=0.7,ecolor='m',label='iminuit',ds='steps-mid')
ax.errorbar(binmid,np.mean(xi1_ELG,axis=0)[3],yerr=np.std(n1array,axis=-1)[3],color='c',alpha=0.7,ecolor='c',label='Multinest',ds='steps-mid')
ax.step(binmid,n,color='k',label='UNIT sim.')
plt.yscale('log')
plt.ylabel('galaxy numbers')
plt.legend(loc=2)
plt.title('Vpeak distribution: {} in {}'.format(gal,GC))
plt.xlabel(var+' (km/s)')
ax.set_xlim(1000,10)

plt.savefig(func+'-MCMC_distr_'+gal+'_'+GC+'.png',bbox_tight=True)
plt.close()
'''

'''
# obs in different zbins
nmu = 120
MU = (np.linspace(0,1,121)[:-1]+np.linspace(0,1,121)[1:]).reshape(1,120)/2+np.zeros((200,120))
for GC in ['NGC','SGC']:
    for zlow,zhigh in zip([0.65,0.6,0.75],[0.8,0.7,1.0]):
        obsfile  = '/global/homes/z/zhaoc/cscratch/EZmock/2PCF/data/2PCF_zbin1/2PCF_eBOSS_LRG_'+GC+'_v7_z'+str(zlow)+'z'+str(zhigh)
        obsout  = home+'2PCF/zbins/'+gal+'_'+GC+'_z'+str(zlow)+'z'+str(zhigh)+'.dat'
        obsz(obsfile,obsout,os.path.exists(obsout))
        zmock = glob.glob('/global/homes/z/zhaoc/cscratch/EZmock/2PCF/LRGv7_comp/z'+str(zlow)+'z'+str(zhigh)+'/2PCF/2PCF_EZmock_eBOSS_LRG_'+GC+'*.dat')
        nfile = len(zmock)
        # read all the 2pcf data
        mockmono = [x for x in range(nfile)]
        mockquadru = [x for x in range(nfile)]
        mockhexadeca = [x for x in range(nfile)]		
        dd = np.zeros((nfile,24000))
        for i in range(nfile):
            dd[i] = Table.read(zmock[i],format='ascii.no_header')['col3']
        dd = dd.reshape(nfile,200,120)
        quad = dd * 2.5 * (3 * MU**2 - 1)
        hexa = dd * 1.125 * (35 * MU**4 - 30 * MU**2 + 3)
        # integral 2pcf with mu to have 2pcf(s)
        mono_tmp = np.trapz(dd, dx=1./nmu, axis=-1)
        quad_tmp = np.trapz(quad, dx=1./nmu, axis=-1)
        hexa_tmp = np.trapz(hexa, dx=1./nmu, axis=-1)
        mockmono  = mono_tmp.T
        mockquad  = np.vstack((mockmono,quad_tmp.T))
        mockhexa  = np.vstack((mockquad,hexa_tmp.T))
        for name,mockarr in zip(['mono','quad','hexa'],[mockmono,mockquad,mockhexa]):
            cols = []
            cols.append(fits.Column(name=name,format=str(nfile)+'D',array=mockarr))
            hdulist = fits.BinTableHDU.from_columns(cols)
            hdulist.writeto(obsout[:-4]+'_'+name+'.fits.gz', overwrite=True)

'''