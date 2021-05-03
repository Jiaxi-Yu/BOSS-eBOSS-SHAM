#!/usr/bin/env python3
import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
import os
import warnings
import matplotlib.pyplot as plt
import glob
import matplotlib.gridspec as gridspec
import sys
import h5py
from halotools.mock_observables import s_mu_tpcf
from halotools.mock_observables import tpcf_multipole
cataname = sys.argv[1]
Vthresh = int(sys.argv[2])
boxsize  = 400
gal      = 'CMASSLOWZTOT'
GC       = 'NGC+SGC'
rscale   = 'linear' # 'log'
function = 'mps' # 'wp'
zmin     = '0.2'
zmax     = '0.75'
nseed    = 30
date     = '0218'
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
rmin     = 5
rmax = 25
nthread  = 16
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=35
home     = '/home/astro/jiayu/Desktop/SHAM/'
cols = ['col4','col5']
finish = False

# start the final 2pcf, wp, Vpeak histogram, PDF
SHAMnum = int(208000/2.5**3)
scathalf = int(SHAMnum/2)
z = 0.5609
a_t = '0.64210'
# generate s bins
bins  = np.arange(rmin,rmax+1,1)
nbins = len(bins)-1
binmin = rmin
binmax = rmax
s = (bins[:-1]+bins[1:])/2
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
halofile = home+'catalog/'+cataname+'_snapshot_009.z0.500.AHF_halos.hdf5'       
read = time.time()
f=h5py.File(halofile,"r")
plt.hist(f["halo"]['Vmax'][:],bins = 71,range=(160,1600))
plt.yscale('log')
plt.savefig('{}_Vmax_hist.png'.format(cataname))
plt.close()
sel = f["halo"]['Vmax'][:]>Vthresh
if len(f["halo"]['Vmax'][sel])%2 ==1:
    datac = np.zeros((len(f["halo"]['Vmax'][sel])-1,5))
    for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = (f["halo"][key][sel])[:-1]
else:
    datac = np.zeros((len(f["halo"]['Vmax'][sel]),5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = f["halo"][key][sel]
f.close()        
datac[:,2:]/=1000
half = int32(len(datac)/2)
print('min values for each colume: ',np.min(datac,axis=0))
print('max values for each colume: ',np.max(datac,axis=0))
print(len(datac))
print('read the halo catalogue costs {:.6}s'.format(time.time()-read))
z_redshift  = datac[:,4]+datac[:,0]*(1+z)/H
z_redshift %=boxsize
datac[:,-1] = z_redshift*1

mu1 = np.linspace(0,mu_max,nmu+1)
xi_s_mu = s_mu_tpcf(datac[:,2:],bins, mu1, period=boxsize, num_threads=nthread)
xi0 = tpcf_multipole(xi_s_mu, mu1, order=0)
xi2 = tpcf_multipole(xi_s_mu, mu1, order=2)
xi = np.array([xi0,xi2]).T
# plot the 2PCF multipoles   
fig= plt.figure(figsize=(14,6))
spec = gridspec.GridSpec(nrows=1,ncols=2)#, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
ax = np.empty((1,2), dtype=type(plt.axes))
for name,k in zip(['monopole','quadrupole'],range(2)):
    values=[np.zeros(nbins),0]        
    for j in range(1):
        ax[j,k] = fig.add_subplot(spec[j,k])
        ax[j,k].plot(s,s**2*(xi[:,k]-values[j]),c='c',alpha=0.6,label='SHAM-python')
        #ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='m',alpha=0.6,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-3)))
        plt.xlabel('s (Mpc $h^{-1}$)')
        if rscale=='log':
            plt.xscale('log')
        if (j==0):
            ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
            if k==0:
                plt.legend(loc=2)
            else:
                plt.legend(loc=1)
            plt.title('correlation function {}: {} in {}'.format(name,gal,GC))


plt.savefig('cftest_{}_Vthresh{}.png'.format(cataname,Vthresh),bbox_tight=True)
plt.close()

