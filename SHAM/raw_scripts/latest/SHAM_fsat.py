import numpy as np
import os
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
import h5py

# variables
gals     = ['LOWZ']*3+['CMASS']*4
zmins    = ['0.2' ,'0.33','0.2', '0.43','0.51','0.57','0.43']
zmaxs    = ['0.33','0.43','0.43','0.51','0.57','0.7' ,'0.7']
GC       = 'NGC+SGC'
rscale   = 'linear' # 'log'
pre      = '/'
date     = '0218'
#'0218': 3-param, '0726':mock-SHAM 3-param, '0729': 2-param
function = 'mps' # 'wp'
nseed    = 32
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
nthread  = 1
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=30
parameters = ["sigma","Vsmear","Vceil"]
rmin     = 5
rmax = 25
cols = ['col4','col5']

home     = '/global/homes/j/jiaxi/SHAM/'
datapath = '/global/cscratch1/sd/jiaxi/SHAM/'

# SHAM halo catalogue
def fsat(datadir,issham=True):
    if issham:
        ID = np.loadtxt(datadir,usecols=3)
    else:
        f=h5py.File(datadir,"r")
        ID =f["halo"]['PID'][:]
        f.close()   
    # calculate the satellite fraction
    return len(ID[ID!=-1])/len(ID)
if not os.path.exists(home+'fsat.txt'):
    print('LOWZ CMASS fsat calculation')
    f = open(home+'fsat.txt','a')
    f.write('# zeff fsat fsat_error UNITfsat\n')
    for gal,zmin,zmax in zip(gals,zmins,zmaxs):
        # start the final 2pcf, wp, Vpeak histogram, PDF
        if gal=='CMASS':
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

        # collect the fsat of all the satellite fraction
        # for SHAM catalogues
        dataroot = datapath+'catalog/fsat/'+'z{}z{}/'.format(zmin,zmax)+'SHAM{}.dat'
        datafile = [dataroot.format(i+1) for i in range(nseed)]
        with Pool(processes = nseed) as p:
            xi1_ELG = p.map(fsat,datafile) 
        # mean and std 
        mean0 = np.mean(np.array(xi1_ELG))
        std0  = np.std(np.array(xi1_ELG))/np.sqrt(nseed)
        # for UNIT catalogues
        fsatUNIT = fsat(datapath+'catalog/UNIT_hdf5/UNIT_hlist_{}_PID.hdf5'.format(a_t),issham=False)
        # write in file
        f.write('# SHAM at {}<z<{} satellite fraction:\n'.format(zmin,zmax))
        f.write('{} {} {} {}\n'.format(z,mean0,std0,fsatUNIT))
    f.close()
else:
    print('plot LOWZ CMASS fsat vs UNIT subhalo fraction')
    data = np.loadtxt(home+'fsat.txt')
    tmp = data[2,:]
    data[2:-2,:] = data[3:-1,:]
    data[-2,:] = tmp
    for k in range(2):
        if k==0:
            z = data[:-2,0]
            fsat = data[:-2,1]
            fsaterr = data[:-2,2]
            fsatUNIT = data[:-2,3]
            name = 'zbins'
        else:
            z = data[-2:,0]
            fsat = data[-2:,1]
            fsaterr = data[-2:,2]
            fsatUNIT = data[-2:,3]
            name = 'bulk'
        
        plt.errorbar(z,fsat*100,fsaterr*100,color='k', marker='o',ecolor='k',ls="none",label='SHAM {}'.format(name)) #,mfc='w'
        plt.scatter(z,fsatUNIT*100,color='k', marker='D',label='UNIT {}'.format(name)) # facecolors='none'
        
        pos = 11.75
        plt.axvline(0.43, color= "k")
        plt.text(0.28, pos, 'LOWZ')
        plt.text(0.53, pos, 'CMASS')
        plt.ylim(11.5,14);plt.xlim(0.2,0.7)
        for i,Z in enumerate(z):
            plt.plot(np.array([Z,Z]),np.array([fsat[i]*100,fsatUNIT[i]*100]),color="b",linestyle='--')
        plt.xlabel('$z_{eff}$')
        plt.ylabel('satellite fraction (%)')
        plt.legend(loc=0)
        plt.savefig(home+'fsat_{}.png'.format(name))
        plt.close()