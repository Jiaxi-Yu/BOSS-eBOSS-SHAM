import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack
from astropy.table import Table
import sys
import matplotlib.gridspec as gridspec 
import astropy.io.fits as fits
from  glob import glob 

gal      = sys.argv[1]
GC       = sys.argv[2]
rscale   = sys.argv[3] #'linear' # 'log'
function = 'mps'
zmin     = sys.argv[4]
zmax     = sys.argv[5]
rmin     = 1
rmax     = 40
multipole= 'hexa' # 'mono','quad','hexa'

home     = '/global/cscratch1/sd/jiaxi/SHAM/'
direc    = '/global/homes/j/jiaxi/'

if (rscale=='linear')&(function=='mps'):
    if gal == 'LRG':
        SHAMnum   = int(6.26e4)
        z = 0.7018
        a_t = '0.58760'
        ver = 'v7_2'
    else:
        SHAMnum   = int(2.93e5)
        z = 0.8594
        a_t = '0.53780'
        ver = 'v7'
       
    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(direc,gal,ver,function,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(direc,gal,ver,function,rscale,gal,multipole)
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:],mock[binmin+400:binmax+400,:]))
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
    if gal == 'LRG':
        obs   = vstack((obscf['col4'],obscf['col5'],obscf['col6']))
    else:
        obs   = vstack((obscf['col3'],obscf['col4'],obscf['col5']))
        
    print('the covariance matrix and the observation 2pcf vector are ready.')
    
elif (rscale=='log'):
    # read s bins
    binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
    bins  = np.unique(np.append(binfile['col1'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)],binfile['col2'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]))
    s = np.array(binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)])
    nbins = len(bins)-1
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    
    if gal == 'LRG':
        ver = 'v7_2'
        extra = binfile['col3'][(binfile['col3']<rmax)&(binfile['col3']>=rmin)]**2
    else:
        ver = 'v7'
        extra = np.ones(binmax-binmin)
    # filenames
    covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(direc,gal,function,rscale,gal,zmin,zmax,multipole) 
    obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(direc,gal,function,rscale,gal,GC,ver,zmin,zmax)

    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1] 
    hdu.close()
    # observations
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance for chi2
    Ns = int(mocks.shape[0]/3)
    mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Ns:binmax+Ns,:],mocks[binmin+2*Ns:binmax+2*Ns,:]))
    obs   = vstack((obscf['col4']/extra,obscf['col5']/extra,obscf['col6']/extra))# LRG columns are s**2*xi
else:
    print('wrong 2pcf function input')

# plot the EZmocks vs observations
median = np.nanmedian(mocks,axis=-1)
disp = np.std(mocks,axis=-1)    
# plot the 2PCF multipoles   
fig = plt.figure(figsize=(21,8))
spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.2,wspace=0.3)
ax = np.empty((2,3), dtype=type(plt.axes))
for name,k in zip(['monopole','quadrapole','hexadecapole'],range(3)):
    j=0
    ax[j,k] = fig.add_subplot(spec[j,k])
    ax[j,k].fill_between(s,s**2*((median-disp)[nbins*k:nbins*(k+1)]),s**2*((median+disp)[nbins*k:nbins*(k+1)]),label='EZmocks 1$\sigma$',color='c')
    ax[j,k].plot(s,s**2*np.nanmedian(mocks[nbins*k:nbins*(k+1)],axis=-1),label='EZmocks median',c='r')
    ax[j,k].plot(s,s**2*obs[k],'k',label='PIP obs')
    plt.xlabel('s (Mpc $h^{-1}$)')
    ax[j,k].set_ylabel('$s^2*\\xi_{}$'.format(k*2))
    plt.legend(loc=0)
    plt.xlim(rmin,rmax)
    plt.title('2PCF {} for {} in {}'.format(name,gal,GC))
    
    # difference
    j=1
    ax[j,k] = fig.add_subplot(spec[j,k])
    ax[j,k].fill_between(s,s**2*((-disp)[nbins*k:nbins*(k+1)]),s**2*((disp)[nbins*k:nbins*(k+1)]),label='EZmocks 1$\sigma$',color='c')
    ax[j,k].plot(s,np.zeros_like(s),label='EZmocks median',c='r')
    ax[j,k].plot(s,s**2*(obs[k]-median[nbins*k:nbins*(k+1)]),'k',label='PIP obs')
    plt.xlabel('s (Mpc $h^{-1}$)')
    plt.xlim(rmin,rmax)
    ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2))

plt.savefig('EZmock_vs_obs_{}_{}_z{}z{}.png'.format(gal,GC,zmin,zmax),bbox_tight=True)
plt.close()
