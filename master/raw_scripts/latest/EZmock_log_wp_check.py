import matplotlib 
matplotlib.use('agg') 
import time
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.table import Table
import astropy.io.fits as fits
import matplotlib.gridspec as gridspec 
import sys


rpmin = 0.1
rpmax = 50
pimax = 80
home = '/global/cscratch1/sd/jiaxi/master/'
for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
    for i,GC in enumerate(['NGC','SGC']):
        # ELG wp from Faizan
        obs = Table.read('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),format='ascii.no_header')
        obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]
        bins = np.unique(np.append(obs['col1'],obs['col2']))
        # EZmocks
        WP = np.loadtxt('{}wp_log_{}_{}.dat'.format(home,gal,GC))
        mean= np.mean(WP,axis=-1)
        disp = np.std(WP,axis=-1)

        # plot obs vs data CP with Corrfunc
        fig = plt.figure(figsize=(7,8))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
        ax = np.empty((2,1), dtype=type(plt.axes))
        for j in range(2):
            ax[j,0] = fig.add_subplot(spec[j,0])
            plt.xlabel('rp (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,0].plot(obs['col3'],mean,label='EZmocks',color='orange')
                ax[j,0].plot(obs['col3'],obs['col4'],label='PIP+ANG',marker='*',color='k')
                ax[j,0].fill_between(obs['col3'],mean-disp,mean+disp,label='EZmocks 1$\sigma$',color='c')
                ax[j,0].set_ylabel('wp($r_p$)') 
                plt.legend(loc=0)
                plt.xscale('log')
                plt.ylim(1,700)
                plt.yscale('log')
                plt.title('projected 2PCF: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,0].scatter(obs['col3'],(mean-obs['col4'])/obs['col4']*100,color='orange')
                plt.xscale('log')
                ax[j,0].plot(obs['col3'],np.ones_like(np.array(obs['col3'])),color='k')
                ax[j,0].set_ylabel('$\Delta$wp(%)')

        plt.savefig('{}wp_{}_{}.png'.format(home,gal,GC),bbox_tight=True)
        plt.close()

