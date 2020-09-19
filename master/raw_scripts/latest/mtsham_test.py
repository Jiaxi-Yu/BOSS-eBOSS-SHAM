import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import time
initial = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,float32,int32,append,mean,cov,vstack,std
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
from multiprocessing import Pool 
from itertools import repeat
import glob
import sys

gal      = 'LRG'
GC       = 'NGC'
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
Om = 0.31
home      = '/global/cscratch1/sd/jiaxi/master/'

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018

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
s = (bins[1:]+bins[:-1])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf
print('reading the halo catalogue for creating the galaxy catalogue...')

# multiprocessing
sig = sys.argv[1]
sigV= sys.argv[2]
Vceil = sys.argv[3]
nseed    = int(sys.argv[4])
if nseed==1:
    extra = 1
else:
    extra=4
if os.path.exists(home+'LRG_NGC-redshift_space_sigma{}_sigmaV{}_Vceil{}_seed{}-python.dat'.format(sig,sigV,Vceil,nseed*extra))==False:
    halofile = home+'catalog/UNIT4LRG-cut200.dat'
    print('reading UNIT catalog')
    data = Table.read(halofile,format='ascii.no_header')

    # make sure len(datac) is even
    if len(data)%2==1:
        data = data[:-1]
    datac = np.zeros((len(data),5))
    for j in range(5):
        datac[:,j] = np.array(data['col{}'.format(j+1)])

    half = int(len(data)/2)
    scathalf = int(SHAMnum/2)

    # generate nseed Gaussian random number arrays in a list
    print('generating uniform random number arrays...')
    uniform_randoms = [np.random.RandomState(seed=x+1).rand(len(datac)).astype('float32') for x in range(nseed)]
    uniform_randoms1 = [np.random.RandomState(seed=x+1+nseed).rand(len(datac)).astype('float32') for x in range(nseed)]
    uniform_randoms2 = [np.random.RandomState(seed=x+1+nseed*2).rand(len(datac)).astype('float32') for x in range(nseed)]
    uniform_randoms3 = [np.random.RandomState(seed=x+1+nseed*3).rand(len(datac)).astype('float32') for x in range(nseed)]
    print('the uniform random number dtype is ',uniform_randoms[0].dtype)

    # HAM application
    def sham_tpcf(uni,uni1,uni2,uni3,sigM,sigV,Mtrun):
        x00,x20,x40,x001,x201,x401=sham_cal(uni,sigM,sigV,Mtrun)
        x01,x21,x41,x011,x211,x411=sham_cal(uni1,sigM,sigV,Mtrun)
        x02,x22,x42,x021,x221,x421=sham_cal(uni2,sigM,sigV,Mtrun)
        x03,x23,x43,x031,x231,x431=sham_cal(uni3,sigM,sigV,Mtrun)
        #return  [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2,(x001+x011)/2,(x201+x211)/2,(x401+x411)/2]
        return  [x00,x01,x02,x03,x20,x21,x22,x23,x42,x43,x40,x41,x001,x011,x021,x031,x201,x211,x221,x231,x401,x411,x421,x431]

    def sham_cal(uniform,sigma_high,sigma,v_high):
        datav = datac[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) 
        LRGscat = (datac[datav<v_high])[argpartition(-datav[datav<v_high],SHAMnum)[:(SHAMnum)]]
        # transfer to the redshift space
        z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
        z_redshift %=boxsize
        # calculate the 2pcf of the SHAM galaxies
        # count the galaxy pairs and normalise them
        DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
        # calculate the 2pcf and the multipoles
        mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
        quad = mono * 2.5 * (3 * mu**2 - 1)
        hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        DD_counts1 = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],LRGscat[:,2],periodic=True, verbose=True,boxsize=boxsize)
        # calculate the 2pcf and the multipoles
        mono1 = (DD_counts1['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
        quad1 = mono1 * 2.5 * (3 * mu**2 - 1)
        hexa1 = mono1 * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
        # use sum to integrate over mu
        return [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,np.sum(hexa,axis=-1)/nmu,np.sum(mono1,axis=-1)/nmu,np.sum(quad1,axis=-1)/nmu,np.sum(hexa1,axis=-1)/nmu]

    if nseed==1:
        xi0_tmp = sham_cal(uniform_randoms[0],np.float32(sig),np.float32(sigV),np.float32(Vceil))
        xi0,xi2,xi4,xi01,xi21,xi41 = (xi0_tmp[0]+xi0_tmp[1])/2,(xi0_tmp[2]+xi0_tmp[3])/2,(xi0_tmp[4]+xi0_tmp[5])/2,(xi0_tmp[6]+xi0_tmp[7])/2,(xi0_tmp[8]+xi0_tmp[9])/2,(xi0_tmp[10]+xi0_tmp[11])/2,
    else:
        with Pool(processes = nseed) as p:
            xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,uniform_randoms2,uniform_randoms3,repeat(np.float32(sig)),repeat(np.float32(sigV)),repeat(np.float32(Vceil))))
            # averages and standard deviations
            xi0,xi2,xi4,xi01,xi21,xi41 = \
            mean([xi0_tmp[k][j] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            mean([xi0_tmp[k][j+4] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            mean([xi0_tmp[k][j+8] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            mean([xi0_tmp[k][j+12] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            mean([xi0_tmp[k][j+16] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            mean([xi0_tmp[k][j+20] for k in range(nseed) for j in range(4)],axis=0,dtype='float32')
            
            xi0std,xi2std,xi4std,xi01std,xi21std,xi41std = \
            std([xi0_tmp[k][j] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            std([xi0_tmp[k][j+4] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            std([xi0_tmp[k][j+8] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            std([xi0_tmp[k][j+12] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            std([xi0_tmp[k][j+16] for k in range(nseed) for j in range(4)],axis=0,dtype='float32'),\
            std([xi0_tmp[k][j+20] for k in range(nseed) for j in range(4)],axis=0,dtype='float32')


    Table([xi0,xi2,xi4,xi0std,xi2std,xi4std]).write(home+'LRG_NGC-redshift_space_sigma{}_sigmaV{}_Vceil{}_seed{}-python.dat'.format(sig,sigV,Vceil,nseed*extra),format = 'ascii.no_header',delimiter='\t',overwrite=True)
    Table([xi01,xi21,xi41,xi01std,xi21std,xi41std]).write(home+'LRG_NGC-real_space_sigma{}_sigmaV{}_Vceil{}_seed{}-python.dat'.format(sig,sigV,Vceil,nseed*extra),format = 'ascii.no_header',delimiter='\t',overwrite=True)

f = open(home+'{}_{}-sigma{}_sigmaV{}_Vceil1e99_seed{}-diff.dat'.format(gal,GC,sig,sigV,nseed*extra),'w')    
for space in ['redshift_space','real_space']:
    python = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil{}_seed{}-python.dat'.format(gal,GC,space,sig,sigV,Vceil,nseed*extra),format='ascii.no_header')
    if Vceil=='1000000':
        c = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil1e99_seed{}-c.dat'.format(gal,GC,space,sig,sigV,nseed*extra),format='ascii.no_header')[binmin:]
    else:
        c = Table.read(home+'{}_{}-{}_sigma{}_sigmaV{}_Vceil{}_seed{}-c.dat'.format(gal,GC,space,sig,sigV,Vceil,nseed*extra),format='ascii.no_header')[binmin:]
    f.write('# mono(%)  quad(%)\n')
    f.write('#{}:\n'.format(space))
    for arr1,arr2 in zip(np.array((c['col3']-python['col1'])/python['col1']*100),np.array((c['col4']-python['col2'])/python['col2']*100)):
        f.write('{} {}'.format(arr1,arr2))
        f.write('\n')


    fig = plt.figure(figsize=(15,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for k in range(2):
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            plt.xlabel('s (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,k].errorbar(s,s**2*python['col{}'.format(k+1)],s**2*python['col{}'.format(k+4)],color='k', marker='.',ms=8,ecolor='k',ls="none",label='python')
                ax[j,k].plot(s,s**2*c['col{}'.format(k+3)],c='c',alpha=0.6,label = 'c')
                ax[j,k].set_ylabel('$\\xi_{}(s)$'.format(k*2)) 
                plt.legend(loc=0)
                plt.title('2PCF: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,k].plot(s,(c['col{}'.format(k+3)]-python['col{}'.format(k+1)])/python['col{}'.format(k+1)]*100,marker='^',color='c')
                ax[j,k].plot(s,np.zeros_like(s),'k--')
                ax[j,k].set_ylabel('$\Delta\\xi_{}$(%)'.format(k*2))

    plt.savefig(home+'{}_{}-{}-sigma{}_sigmaV{}_Vceil{}_seed{}-diff.png'.format(gal,GC,space,sig,sigV,Vceil,nseed*extra),bbox_tight=True)
    plt.close()
