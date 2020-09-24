import matplotlib 
matplotlib.use('agg') 
import time
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.table import Table
import astropy.io.fits as fits
from astropy.io import ascii 
from astropy.cosmology import FlatLambdaCDM
import matplotlib.gridspec as gridspec 
import sys


rpmin = 0.1
rpmax =50
pimax = 80
#for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
#gal = 'LRG'
#ver = 'v7_2'
gal = 'ELG'
ver = 'v7'
home = '/global/cscratch1/sd/jiaxi/master/'
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=2.725)
weights1,weights2=[0,1],[0,1]
ra,dec,dcom,ra1,dec1,dcom1=[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]

for i,GC in enumerate(['NGC','SGC']):
    # ELG wp from Faizan
    obs = ascii.read('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),format='no_header')
    obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]
    bins = np.unique(np.append(obs['col1'],obs['col2']))

    # ELG clustering obs
    hdu=fits.open('{}catalog/eBOSS_{}_clustering_{}_{}.dat.fits'.format(home,gal,GC,ver)) 
    data = hdu[1].data
    data = data[~np.isnan(data['RA'])]
    hdu.close()
    ra[i],dec[i],z = data['RA'],data['DEC'],data['Z']
    weights1[i] =data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*data['WEIGHT_CP']*data['WEIGHT_NOZ']
    data=np.zeros(0)
    # ELG clustering random
    hdu=fits.open('{}catalog/eBOSS_{}_clustering_{}_{}.ran.fits'.format(home,gal,GC,ver)) 
    data = hdu[1].data
    data = data[~np.isnan(data['RA'])]
    hdu.close()
    ra1[i],dec1[i],z1 = data['RA'],data['DEC'],data['Z'] 
    weights2[i] =data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*data['WEIGHT_CP']*data['WEIGHT_NOZ']
    print('random&obs ready')
    #data=np.zeros(0)
    dcom[i] = cosmo.comoving_distance(np.array(z))*67.7/100
    dcom1[i] = cosmo.comoving_distance(np.array(z1))*67.7/100
    print('z->dcom finished')

    catalog = [x for x in range(8)]
    for k,arr in enumerate([ra,dec,dcom,weights1,ra1,dec1,dcom1,weights2]):
        catalog[k] = arr[i]#,arr[1])

    a=time.time()
    '''
    # halotools: failed wp
    from halotools.mock_observables import wp
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    # 
    XYZ = SkyCoord(ra=catalog[0]*u.degree, dec=catalog[1]*u.degree, distance=catalog[2]*u.mpc)
    XYZ1 = SkyCoord(ra=catalog[4]*u.degree, dec=catalog[5]*u.degree, distance=catalog[6]*u.mpc)
    coords = np.vstack((XYZ.cartesian.x,XYZ.cartesian.y,XYZ.cartesian.z)).T
    coords1 = np.vstack((XYZ1.cartesian.x,XYZ1.cartesian.y,XYZ1.cartesian.z)).T
    WP = wp(coords,bins,pimax, randoms=coords1,num_threads=64)
    '''
    '''
    #Corrfunc: failed wp
    from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks 
    from Corrfunc.utils import convert_rp_pi_counts_to_wp 
    from Corrfunc.utils import convert_3d_counts_to_cf
    DD = DDrppi_mocks(1, 2, 64, pimax,bins,catalog[0],catalog[1],catalog[2],weights1=catalog[3],is_comoving_dist=True,weight_type="pair_product")
    DR = DDrppi_mocks(0, 2, 64, pimax,bins,catalog[0],catalog[1],catalog[2],weights1=catalog[3],is_comoving_dist=True, RA2=catalog[4], DEC2=catalog[5], CZ2=catalog[6],weights2=catalog[7],weight_type="pair_product")
    RR = DDrppi_mocks(1, 2, 64, pimax,bins,catalog[4],catalog[5],catalog[6],weights1=catalog[7],is_comoving_dist=True,weight_type="pair_product")
    # projected 2D:
    WP = convert_rp_pi_counts_to_wp(len(z), len(z), len(z1), len(z1),DD, DR,DR,RR,len(bins)-1,pimax)
    # or 3D->2D
    CF = convert_3d_counts_to_cf(len(z), len(z), len(z1), len(z1),DD, DR,DR,RR)
    WP = 2*np.sum(CF.reshape(len(obs),pimax),axis=1) 
    '''
    CF = Table.read('{}CUTE/CUTE/wp/log_{}_{}.dat'.format(home[:-7],gal,GC),format='ascii.no_header')['col3']
    WP = 2*np.sum(CF.reshape(len(obs),pimax),axis=1) 
    b=time.time()
    print('Corrfunc used {} s to calculate wp'.format(b-a))

    # plot obs vs data CP with Corrfunc
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    for j in range(2):
        ax[j,0] = fig.add_subplot(spec[j,0])
        plt.xlabel('rp (Mpc $h^{-1}$)')
        if (j==0):
            ax[j,0].scatter(obs['col3'],WP,label='data CP',marker='^',color='orange')
            ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='*',color='k')
            ax[j,0].set_ylabel('wp(r_p)') 
            plt.legend(loc=0)
            plt.xscale('log')
            plt.ylim(1,700)
            plt.yscale('log')
            plt.title('projected 2PCF: {} in {}'.format(gal,GC))
        if (j==1):
            ax[j,0].scatter(obs['col3'],(WP-obs['col4'])/obs['col4']*100,label='data CP',marker='^',color='orange')
            plt.xscale('log')
            ax[j,0].scatter(obs['col3'],np.ones_like(np.array(obs['col3'])),label='PIP+ANG',marker='*',color='k')
            ax[j,0].set_ylabel('$\Delta$wp(%)')

    plt.savefig('{}wp_{}_{}.png'.format(home[:-7],gal,GC),bbox_tight=True)
    plt.close()

    
