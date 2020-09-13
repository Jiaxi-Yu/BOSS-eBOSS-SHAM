import matplotlib 
matplotlib.use('agg') 
import time
import matplotlib.pyplot as plt 
import numpy as np 
from os.path import dirname, abspath, join as pjoin
import Corrfunc 
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks 
from Corrfunc.io import read_catalog 
from Corrfunc.utils import convert_rp_pi_counts_to_wp 
import astropy.io.fits as fits
from astropy.io import ascii 
from astropy.cosmology import FlatLambdaCDM
import matplotlib.gridspec as gridspec 

for GC in ['NGC','SGC']:
    # ELG wp from Faizan
    obs = ascii.read('./catalog/nersc_wp_ELG_v7/wp-pip_eBOSS_ELG_{}_v7.dat'.format(GC),format='no_header')
    bins = np.unique(np.append(obs['col1'],obs['col2']))
    # ELG clustering obs
    hdu=fits.open('./catalog/eBOSS_ELG_clustering_{}_v7.dat.fits'.format(GC)) 
    data = hdu[1].data
    hdu.close()
    ra,dec,z = data['RA'],data['DEC'],data['Z']
    weight =data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*data['WEIGHT_CP']*data['WEIGHT_NOZ']
    # ELG clustering random(error?)
    hdu=fits.open('./catalog/eBOSS_ELG_clustering_{}_v7.ran.fits'.format(GC)) 
    data = hdu[1].data
    hdu.close()
    ra1,dec1,z1 = data['RA'][~np.isnan(data['RA'])],data['DEC'][~np.isnan(data['RA'])] ,data['Z'][~np.isnan(data['RA'])] 
   
    # z -> dcom
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=2.725)
    dcom = cosmo.comoving_distance(np.array(z))*67.7/100
    dcom1 = cosmo.comoving_distance(np.array(z1))*67.7/100

    a=time.time()
    DD_counts = DDrppi_mocks(1, 2, 64, 80,bins,ra,dec,dcom,weights1=weight,is_comoving_dist=True)
    DR_counts = DDrppi_mocks(0, 2, 64, 80,bins,ra,dec,dcom,weights1=weight,is_comoving_dist=True, RA2=ra1, DEC2=dec1, CZ2=dcom1)
    RR_counts = DDrppi_mocks(1, 2, 64, 80,bins,ra1,dec1,dcom1,is_comoving_dist=True)
    wp = convert_rp_pi_counts_to_wp(len(z), len(z), len(z1), len(z1),DD_counts, DR_counts,DR_counts, RR_counts,len(bins)-1,80)
    b=time.time()
    print('Corrfunc used {} s to calculate wp'.format(b-a))
    
    # plot obs vs data CP with Corrfunc
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    values=[np.zeros(nbins),obscf['col3']]
    for j in range(2):
        ax[j,0] = fig.add_subplot(spec[j,0])
        plt.xlabel('s (Mpc $h^{-1}$)')
        if (j==0):
            ax[j,0].scatter(obs['col3'],wp,label='data CP',marker='^')
            ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='*')
            ax[j,0].set_ylabel('$\\wp(r_p)$') 
            plt.legend(loc=0)
            plt.xscale('log')
            plt.ylim(1,700)
            plt.yscale('log')
            plt.title('projected 2PCF: {} in {}'.format(gal,GC))
        if (j==1):
            ax[j,0].scatter(obs['col3'],(wp-obs['col4'])/obs['col4']*100,label='data CP',marker='^')
            plt.xscale('log')
            ax[j,0].scatter(obs['col3'],np.ones_like(np.array(obs['col3'])),label='PIP+ANG',marker='*')
            ax[j,k].set_ylabel('$\Delta\\wp$(%)')

    plt.savefig('wp-'+gal+'_'+GC+'.png',bbox_tight=True)
    plt.close()
