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

gal      = 'LOWZ'
GC       = sys.argv[1]
rscale   = 'linear' # 'log'
function = 'mps'
zmin     = sys.argv[2]
zmax     = sys.argv[3]
rmin     = 5
rmax = 25
multipole= 'quad' # 'mono','quad','hexa'
pre = '/'
home     = '/home/astro/jiayu/Desktop/SHAM/'
fileroot = '{}MCMCout/zbins_0218/{}{}_{}_{}_{}_z{}z{}/best-fit_{}_{}.dat'.format(home,pre,function,rscale,gal,'NGC+SGC',zmin,zmax,gal,'NGC+SGC')

if (rscale=='linear')&(function=='mps'):
    if gal == 'LRG':
        SHAMnum   = int(6.26e4)
        z = 0.7781
        a_t = '0.56220'
        ver = 'v7_2'
    elif gal=='ELG':
        SHAMnum   = int(2.93e5)
        z = 0.87364
        a_t = '0.53780'
        ver = 'v7'
    elif gal=='CMASSLOWZTOT':
        SHAMnum = 208000
        z = 0.5609
        a_t = '0.64210'
    elif gal=='CMASS':
        if (zmin=='0.43')&(zmax=='0.51'): 
            SHAMnum = 342000
            z = 0.4686
            a_t = '0.68620'
        elif zmin=='0.51':
            SHAMnum = 363000
            z = 0.5417 
            a_t = '0.64210'
        elif zmin=='0.57':
            SHAMnum = 160000
            z = 0.6399
            a_t =  '0.61420'
        elif (zmin=='0.43')&(zmax=='0.7'):            
            SHAMnum = 264000
            z = 0.5897
            a_t = '0.62800'
    elif gal=='LOWZ':
        if (zmin=='0.2')&(zmax=='0.33'):            
            SHAMnum = 337000
            z = 0.2754
            a_t = '0.78370' 
        elif zmin=='0.33':
            SHAMnum = 258000
            z = 0.3865
            a_t = '0.71730'
        elif (zmin=='0.2')&(zmax=='0.43'): 
            SHAMnum = 295000
            z = 0.3441
            a_t = '0.74980'
    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    if (gal == 'LRG')|(gal=='ELG'):
        obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
        covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
    else:
        obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_{}_DR12v5_z{}z{}.mps'.format(home,gal,GC,zmin,zmax)
        covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
    
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    if (gal == 'LRG')|(gal=='ELG'):
        Nstot=200
    else:
        Nstot=100
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
    covcut  = cov(mocks).astype('float32')
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
    if gal == 'ELG':
        OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
    else:
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    print('the covariance matrix and the observation 2pcf vector are ready.')
elif (rscale=='log'):
    # read s bins
    binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header');ver1='v7_2'
    sel = (binfile['col3']<rmax)&(binfile['col3']>=rmin)
    bins  = np.unique(np.append(binfile['col1'][sel],binfile['col2'][sel]))
    s = binfile['col3'][sel]
    nbins = len(bins)-1
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1

    if gal == 'LRG':
        ver = 'v7_2'
    else:
        ver = 'v7'
    # filenames
    covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
    obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mocks = hdu[1].data[GC+'mocks']
    Nmock = mocks.shape[1]
    hdu.close()
    # observations
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance and errobar for chi2
    Nstot = int(mocks.shape[0]/2)
    mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Nstot:binmax+Nstot,:]))
    covcut  = cov(mocks).astype('float32')
    OBS   = append(obscf['col4'],obscf['col5']).astype('float32')# LRG columns are s**2*xi
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    print('the covariance matrix and the observation 2pcf vector are ready.')

    # zbins, z_eff ans ngal
    if (zmin=='0.6')&(zmax=='0.8'):
        if gal=='ELG':
            SHAMnum = int(3.26e5)
            z = 0.7136
        else:
            SHAMnum = int(8.86e4)
            z = 0.7051
        a_t = '0.58760'
    elif (zmin=='0.6')&(zmax=='0.7'):            
        SHAMnum = int(9.39e4)
        z = 0.6518
        a_t = '0.60080'
    elif zmin=='0.65':
        SHAMnum = int(8.80e4)
        z = 0.7273
        a_t = '0.57470'
    elif zmin=='0.9':
        SHAMnum = int(1.54e5)
        z = 0.9938
        a_t = '0.50320'
    elif zmin=='0.7':
        if gal=='ELG':
            SHAMnum = int(4.38e5)
            z = 0.8045# To be calculated
        else:
            SHAMnum = int(6.47e4)
            z=0.7968
        a_t = '0.54980'
    else:
        if gal=='ELG':
            SHAMnum = int(3.34e5)
            z = 0.9045 # To be calculated
        else:
            SHAMnum = int(3.01e4)
            z= 0.8777
        a_t = '0.52600'
else:
    print('wrong 2pcf function input')

if rscale == 'linear':
    Ccode = np.loadtxt(fileroot)[binmin:binmax]
    Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[binmin:binmax]
else:
    Ccode = np.loadtxt(fileroot)[1:]
    Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[1:]

disp = np.std(mocks,axis=1)

fig = plt.figure(figsize=(14,8))
spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.2,wspace=0.2)
ax = np.empty((2,2), dtype=type(plt.axes))
for name,k in zip(['monopole','quadrupole'],range(2)):
    values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
    err   = [np.ones(nbins),s**2*disp[k*nbins:(k+1)*nbins]]
    for j in range(2):
        ax[j,k] = fig.add_subplot(spec[j,k])
        # mocks mean and std
        ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='c',label='SHAM {} Vsmear best'.format(GC))
        ax[j,k].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='m',label='SHAM {} $\Delta v$ best'.format(GC))
        ax[j,k].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs {}'.format(GC))

        plt.xlabel('s (Mpc $h^{-1}$)')
        if rscale=='log':
            plt.xscale('log')
        if (j==0):
            ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
            if k==0:
                pass
                #plt.legend(loc=2)
            else:
                plt.legend(loc=1)
            plt.title('correlation function {} in {} at {}<z<{}'.format(name,GC,zmin,zmax))
        if (j==1):
            ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2))
            plt.ylim(-3,3)

plt.savefig('{}_{}_z{}z{}_s{}-{}Mpch-1-quadtest.png'.format(gal,GC,zmin,zmax,rmin,rmax),bbox_tight=True)
plt.close()

fig = plt.figure(figsize=(14,8))
spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.2,wspace=0.2)
ax = np.empty((2,2), dtype=type(plt.axes))
for name,k in zip(['monopole','quadrupole'],range(2)):
    values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
    err   = [np.ones(nbins),np.ones(nbins)]
    for j in range(2):
        ax[j,k] = fig.add_subplot(spec[j,k])
        # mocks mean and std
        ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='c',label='SHAM {} Vsmear best'.format(GC))
        ax[j,k].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='m',label='SHAM {} $\Delta v$ best'.format(GC))
        ax[j,k].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs {}'.format(GC))

        plt.xlabel('s (Mpc $h^{-1}$)')
        if rscale=='log':
            plt.xscale('log')
        if (j==0):
            ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
            if k==0:
                pass
                #plt.legend(loc=2)
            else:
                plt.legend(loc=1)
            plt.title('correlation function {} in {} at {}<z<{}'.format(name,GC,zmin,zmax))
        if (j==1):
            ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2))
            plt.ylim(-15,5)

plt.savefig('{}_{}_z{}z{}_s{}-{}Mpch-1-quadtest-absdiff.png'.format(gal,GC,zmin,zmax,rmin,rmax),bbox_tight=True)
plt.close()
