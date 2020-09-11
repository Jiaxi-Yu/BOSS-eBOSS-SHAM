import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from astropy.io import ascii
from NGC_SGC import read_xi
import matplotlib.gridspec as gridspec 
import astropy.io.fits as fits

gal = sys.argv[1]
zmin = sys.argv[2]
zmax = sys.argv[3]
num=sys.argv[4]
s =ascii.read('/global/cscratch1/sd/jiaxi/master/binfile_log_80.dat',format = 'no_header')['col3']

if gal == 'LRG':
    ver='v7_2'
else:
    ver='v7'
    
if len(zmin)==3:
    obs =ascii.read('/global/cscratch1/sd/jiaxi/master/catalog/nersc_zbins_wp_mps_{}/\
mps_log_{}_NGC+SGC_eBOSS_{}_zs_{}0-{}0.dat'.format(gal,gal,ver,zmin,zmax),format = 'no_header')[:len(s)]
else:
    obs =ascii.read('/global/cscratch1/sd/jiaxi/master/catalog/nersc_zbins_wp_mps_{}/\
mps_log_{}_NGC+SGC_eBOSS_{}_zs_{}-{}0.dat'.format(gal,gal,ver,zmin,zmax),format = 'no_header')[:len(s)]
    

sfalse,xi0,xi2,xi4 = read_xi('/global/cscratch1/sd/jiaxi/master/catalog/nersc_zbins_wp_mps_'+gal+'/EZmocks/z'+zmin+'z'+zmax+'/2PCF/2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+zmin+'z'+zmax+'_'+num+'.{}','/global/cscratch1/sd/jiaxi/master/catalog/nersc_zbins_wp_mps_'+gal+'/EZmocks/z'+zmin+'z'+zmax+'/2PCF/2PCF_EZmock_eBOSS_'+gal+'_{}_v7_z'+zmin+'z'+zmax+'.rr',ns=30)

hdu = fits.open('/global/cscratch1/sd/jiaxi/master/catalog/nersc_mps_{}_{}/2PCF_mps_linear_{}_mocks_hexa.fits.gz'.format(gal,ver,gal))
cov = hdu[1].data['NGC+SGCmocks'][:,int(num)-1]
slin = (np.arange(0,200,1)+ np.arange(1,201,1))/2
                             
fig = plt.figure(figsize=(21,8))
spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
ax = np.empty((2,3), dtype=type(plt.axes))
for col,data,name,k in zip(['col4','col5','col6'],[xi0[2],xi2[2],xi4[2]],['mono','quad','hexa'],range(3)):
    j=0
    ax[j,k] = fig.add_subplot(spec[j,k])
    ax[j,k].plot(slin,slin**2*cov[200*k:200*(k+1)],'k--',alpha=0.6,label='mock linear')
    '''
    if gal=='LRG':
        ax[j,k].plot(s,obs[col],'k--',alpha=0.6,label='Faizan')
    else:
        ax[j,k].plot(s,s**2*obs[col],'k--',alpha=0.6,label='Faizan')
    '''
    ax[j,k].plot(s,s**2*data,c='r',alpha=0.6,label='mock log')
    plt.xlabel('s (Mpc $h^{-1}$)')
    ax[j,k].set_ylabel('$s^2*\\xi_{}$'.format(k*2))
    plt.legend(loc=0)
    plt.xscale('log')
    plt.xlim(0.01,100)
    plt.title('correlation function {}: {}'.format(name,gal))
    j=1
    ax[j,k] = fig.add_subplot(spec[j,k])
    '''
    ax[j,k].plot(s,np.ones_like(s),'k--',alpha=0.6,label='Faizan')
    if gal=='LRG':
        ax[j,k].plot(s,(s**2*data/obs[col]-1)*100,c='r',alpha=0.6,label='2pcf code')
    else:
        ax[j,k].plot(s,(data/obs[col]-1)*100,c='r',alpha=0.6,label='2pcf code')
    '''
    plt.xlabel('s (Mpc $h^{-1}$)')
    ax[j,k].set_ylabel('$\Delta\\xi_{}$(%)'.format(k*2))
    plt.xscale('log')
    plt.xlim(0.01,100)
    plt.title('correlation function {}: {}'.format(name,gal))

plt.savefig('2pcf_comparison_{}_z{}z{}_{}.png'.format(gal,zmin,zmax,num),bbox_tight=True)
plt.close()