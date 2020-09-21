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
import sys

rpmin = 0.1
rpmax = 30
pimax = 80
#for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
#gal = 'LRG'
#ver = 'v7_2'
gal = 'ELG'
ver = 'v7'
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=2.725)
weights1,weights2=[0,1],[0,1]
ra,dec,dcom,ra1,dec1,dcom1=[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]

# ELG wp from Faizan
obs = ascii.read('./catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(gal,ver,gal,'NGC',ver),format='no_header')
obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]
bins = np.unique(np.append(obs['col1'],obs['col2']))

for i,GC in enumerate(['NGC','SGC']):
    # ELG clustering obs
    hdu=fits.open('./catalog/eBOSS_{}_clustering_{}_{}.dat.fits'.format(gal,GC,ver)) 
    data = hdu[1].data
    hdu.close()
    ra[i],dec[i],z = data['RA'],data['DEC'],data['Z']
    weights1[i] =data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*data['WEIGHT_CP']*data['WEIGHT_NOZ']
    # ELG clustering random
    hdu=fits.open('./catalog/eBOSS_{}_clustering_{}_{}.ran.fits'.format(gal,GC,ver)) 
    data = hdu[1].data
    hdu.close()
    ra1[i],dec1[i],z1 = data['RA'][~np.isnan(data['RA'])],data['DEC'][~np.isnan(data['RA'])] ,data['Z'][~np.isnan(data['RA'])] 
    weights2[i] =data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*data['WEIGHT_CP']*data['WEIGHT_NOZ']
    print('random&obs ready')

    # z -> dcom
    dcom[i] = cosmo.comoving_distance(np.array(z))*67.7/100
    dcom1[i] = cosmo.comoving_distance(np.array(z1))*67.7/100
    print('z->dcom finished')

catalog = [x for x in range(8)]
for k,arr in enumerate([ra,dec,dcom,weights1,ra1,dec1,dcom1,weights2]):
    catalog[k] = np.append(arr[0],arr[1])
    

a=time.time() 
DD = DDrppi_mocks(1, 2, 64, pimax,bins,catalog[0],catalog[1],catalog[2],weights1=catalog[3],is_comoving_dist=True,weight_type="pair_product")
DR = DDrppi_mocks(0, 2, 64, pimax,bins,catalog[0],catalog[1],catalog[2],weights1=catalog[3],is_comoving_dist=True, RA2=catalog[4], DEC2=catalog[5], CZ2=catalog[6],weights2=catalog[7],weight_type="pair_product")
RR = DDrppi_mocks(1, 2, 64, pimax,bins,catalog[4],catalog[5],catalog[6],weights1=catalog[7],is_comoving_dist=True,weight_type="pair_product")
wp = convert_rp_pi_counts_to_wp(len(z), len(z), len(z1), len(z1),DD, DR,DR,RR,len(bins)-1,pimax)
b=time.time()
print('Corrfunc used {} s to calculate wp'.format(b-a))

# plot obs vs data CP with Corrfunc
fig = plt.figure(figsize=(7,8))
spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for j in range(2):
    ax[j,0] = fig.add_subplot(spec[j,0])
    plt.xlabel('s (Mpc $h^{-1}$)')
    if (j==0):
        ax[j,0].scatter(obs['col3'],wp,label='data CP',marker='^',color='orange')
        #ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='*',color='k')
        ax[j,0].set_ylabel('$\\wp(r_p)$') 
        plt.legend(loc=0)
        plt.xscale('log')
        plt.ylim(1,700)
        plt.yscale('log')
        plt.title('projected 2PCF: {} in {}'.format(gal,GC))
    if (j==1):
        #ax[j,0].scatter(obs['col3'],(wp-obs['col4'])/obs['col4']*100,label='data CP',marker='^',color='orange')
        plt.xscale('log')
        ax[j,0].scatter(obs['col3'],np.ones_like(np.array(obs['col3'])),label='PIP+ANG',marker='*',color='k')
        ax[j,0].set_ylabel('$\Delta\\wp$(%)')

plt.savefig('wp-'+gal+'_NGC+SGC.png',bbox_tight=True)
plt.close()
'''
# Corrfunc.theory
from Corrfunc.theory.wp wp
hdu = fits.open('/global/cscratch1/sd/jiaxi/master/catalog/UNIT_hlist_0.53780.fits.gz')
data = hdu[1].data

result1 = wp(1, 2, 64, pimax,bins,data['X'],data['Y'],data['Z'],)
'''exit
