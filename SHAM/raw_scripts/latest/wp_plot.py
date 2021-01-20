import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib.gridspec as gridspec

#gal='ELG'
GC = 'NGC'
home = '/global/cscratch1/sd/jiaxi/master/'
rpmin=0.1
rpmax=80
#for GC in ['NGC','SGC']:
for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
    A = Table.read('wp/log_{}_{}_rp80.dat'.format(gal,GC),format='ascii.no_header')
    obs = Table.read('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),format='ascii.no_header')
    obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]
    
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    for j in range(2):
        ax[j,0] = fig.add_subplot(spec[j,0])
        plt.xlabel('rp (Mpc $h^{-1}$)')
        if (j==0):
            ax[j,0].scatter(obs['col3'],2*np.sum(A['col3'].reshape(len(obs),80),axis=1),label='data CP',marker='^',color='orange')
            ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='*',color='k')
            ax[j,0].set_ylabel('wp$(r_p)$')
            plt.legend(loc=0)
            plt.xscale('log')
            plt.ylim(1,700)
            plt.yscale('log')
            plt.title('projected 2PCF: {} in {}'.format(gal,GC))
        if (j==1):
            ax[j,0].scatter(obs['col3'],(2*np.sum(A['col3'].reshape(len(obs),80),axis=1)-obs['col4'])/obs['col4']*100,label='data CP',marker='^',color='orange')
            plt.xscale('log')
            ax[j,0].scatter(obs['col3'],np.ones_like(np.array(obs['col3'])),label='PIP+ANG',marker='*',color='k')
            ax[j,0].set_ylabel('$\Delta$wp(%)')

    plt.savefig('wp-{}_{}_rp80.png'.format(gal,GC),bbox_tight=True)
    plt.close()

