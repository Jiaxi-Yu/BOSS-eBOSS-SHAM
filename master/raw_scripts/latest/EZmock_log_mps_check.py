import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from astropy.io import ascii
from NGC_SGC import read_xi
import matplotlib.gridspec as gridspec 
import astropy.io.fits as fits
from  glob import glob 

gal = sys.argv[1]
zmin = sys.argv[2]
zmax = sys.argv[3]
rmin = int(sys.argv[4])
rmax = int(sys.argv[5])

home = '/global/cscratch1/sd/jiaxi/master/'
files = glob(home+'catalog/nersc_zbins_wp_mps_'+gal+'/EZmocks/z'+zmin+'z'+zmax+'/2PCF/2PCF_EZmock_eBOSS_'+gal+'_NGC_v7_z'+zmin+'z'+zmax+'_*.dd') 

s =ascii.read(home+'binfile_log.dat',format = 'no_header')['col3']

if gal == 'LRG':
    ver='v7_2'
else:
    ver='v7'

obs =ascii.read(home+'catalog/nersc_zbins_wp_mps_{}/\
mps_log_{}_NGC+SGC_eBOSS_{}_zs_{}-{}.dat'.format(gal,gal,ver,zmin,zmax),format = 'no_header')
    
if (gal=='ELG')&(sys.argv[6]=='1'):
        # compare ELG obs with pair counts from Faizan
    pairs = ascii.read(home+'catalog/PIP+ANG_pair_counts/pairs_s-mu_log_eBOSS_ELG_NGC+SGC_v7_pip_zs_{}0-{}0.dat'.format(zmin,zmax))
    mu = (np.linspace(0,1,201)[1:]+np.linspace(0,1,201)[:-1])/2
    mask = (pairs['col5']==0)
    mon = np.zeros_like(pairs['col3'])
    mon[~mask]=((pairs['col3'][~mask]-2*pairs['col4'][~mask]+pairs['col5'][~mask])/pairs['col5'][~mask])
    mon = mon.reshape(33,200)
    qua = mon * 2.5 * (3 * mu**2 - 1)
    hexad = mon * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    ## use trapz to integrate over mu
    #** modification made
    obs0 = np.sum(mon,axis=1)/200.
    obs1 = np.sum(qua,axis=1)/200.
    obs2 = np.sum(hexad,axis=1)/200.
    print('monopole are the same? :',max(abs((obs0-obs['col4']))))
    print('quadrupole diff:',max(abs((obs1-obs['col5']))))
    print('hexadecapole diff:',max(abs((obs2-obs['col6']))))

obs = obs[(obs['col3']>=min(s))&(obs['col3']<=max(s))]  

# linear EZmock results
#hdu = fits.open('/global/cscratch1/sd/jiaxi/master/catalog/nersc_mps_{}_{}/2PCF_mps_linear_{}_mocks_hexa.fits.gz'.format(gal,ver,gal))
#cov = hdu[1].data['NGC+SGCmocks'][:,int(num)-1]
#slin = (np.arange(0,200,1)+ np.arange(1,201,1))/2


fig = plt.figure(figsize=(21,8))
spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
ax = np.empty((2,3), dtype=type(plt.axes))
if len(files)<20:
    for l in range(len(files)):
        num = ('{}'.format(l+1)).zfill(4)
        sfalse,xi0,xi2,xi4 = read_xi(home+'catalog/nersc_zbins_wp_mps_'+gal+'/EZmocks/z'+zmin+'z'+zmax+'/2PCF/2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+zmin+'z'+zmax+'_'+num+'.{}',home+'catalog/nersc_zbins_wp_mps_'+gal+'/EZmocks/z'+zmin+'z'+zmax+'/2PCF/2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+zmin+'z'+zmax+'.rr',ns=11)
        for col,data,name,k in zip(['col4','col5','col6'],[xi0[2],xi2[2],xi4[2]],['mono','quad','hexa'],range(3)):
            j=0
            ax[j,k] = fig.add_subplot(spec[j,k])
            #ax[j,k].plot(slin,slin**2*cov[200*k:200*(k+1)],'k--',alpha=0.6,label='mock linear')
            if l==0:
                if gal=='LRG':
                    ax[j,k].plot(s,obs[col],'k--',alpha=0.6,label='obs')
                else:
                    ax[j,k].plot(s,s**2*obs[col],'k--',alpha=0.6,label='obs')

            ax[j,k].plot(s,s**2*data,c='r',alpha=0.6,label='EZmock_{}'.format(num))
            plt.xlabel('s (Mpc $h^{-1}$)')
            ax[j,k].set_ylabel('$s^2*\\xi_{}$'.format(k*2))
            plt.legend(loc=0)
            plt.xlim(rmin,rmax)
            plt.title('correlation function {}: {}'.format(name,gal))
            j=1
            ax[j,k] = fig.add_subplot(spec[j,k])
            if l==0:
                ax[j,k].plot(s,np.ones_like(s),'k--',alpha=0.6)
            if gal=='LRG':
                ax[j,k].plot(s,(s**2*data/obs[col]-1)*100,c='r',alpha=0.6)
            else:
                ax[j,k].plot(s,(data/obs[col]-1)*100,c='r',alpha=0.6)

            plt.xlabel('s (Mpc $h^{-1}$)')
            ax[j,k].set_ylabel('$\Delta\\xi_{}$(%)'.format(k*2))
            plt.xlim(rmin,rmax)
            plt.title('correlation function {}: {}'.format(name,gal))
else:
    hdu = fits.open('{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,'mps','log',gal,zmin,zmax,'hexa'))
    mocks = hdu[1].data['NGC+SGCmocks']
    Ns = int(mocks.shape[0]/3)
    median = np.nanmedian(mocks,axis=-1)
    disp = np.std(mocks,axis=-1)
    hdu.close()
    for col,name,k in zip(['col4','col5','col6'],['mono','quad','hexa'],range(3)):
        j=0
        ax[j,k] = fig.add_subplot(spec[j,k])
        ax[j,k].fill_between(s,s**2*((median-disp)[Ns*k:Ns*(k+1)]),s**2*((median+disp)[Ns*k:Ns*(k+1)]),label='EZmocks 1$\sigma$',color='c')
        ax[j,k].plot(s,s**2*np.nanmedian(mocks[Ns*k:Ns*(k+1)],axis=-1),label='EZmocks median',c='r')
        if gal=='LRG':
            ax[j,k].plot(s,obs[col],'k',label='obs')
        else:
            ax[j,k].plot(s,s**2*obs[col],'k',label='obs')
        plt.xlabel('s (Mpc $h^{-1}$)')
        ax[j,k].set_ylabel('$s^2*\\xi_{}$'.format(k*2))
        plt.legend(loc=0)
        plt.xlim(rmin,rmax)
        plt.title('correlation function {}: {}'.format(name,gal))
        j=1
        ax[j,k] = fig.add_subplot(spec[j,k])
        ax[j,k].plot(s,np.ones_like(s),'k--',alpha=0.6)
        if gal=='LRG':
            ax[j,k].plot(s,(s**2*np.nanmedian(mocks[Ns*k:Ns*(k+1)],axis=-1)/obs[col]-1)*100,c='r',alpha=0.6)
        else:
            ax[j,k].plot(s,(np.nanmedian(mocks[Ns*k:Ns*(k+1)],axis=-1)/obs[col]-1)*100,c='r',alpha=0.6)

        plt.xlim(rmin,rmax)
        plt.xlabel('s (Mpc $h^{-1}$)')
        ax[j,k].set_ylabel('$\Delta\\xi_{}$(%)'.format(k*2))
        plt.title('correlation function {}: {}'.format(name,gal))

    
plt.savefig('2pcf_comparison_{}_z{}z{}.png'.format(gal,zmin,zmax),bbox_tight=True)
plt.close()
