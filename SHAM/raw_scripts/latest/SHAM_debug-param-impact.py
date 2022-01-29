import numpy as np
import os
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
import h5py
import sys

paramtype = sys.argv[1]
# 'free-param' 'Ngal'
if paramtype == 'free-param':
    home  = '/global/cscratch1/sd/jiaxi/SHAM/'
    # variables
    date     = '0218'
    output     = '/global/homes/j/jiaxi/SHAM'
    fileroot = '{}MCMCout/zbins_{}/param_impact/'.format(output,date)
    cols = ['col2','col3']
    parameters = ["sigma","Vsmear","Vceil"]
    rmin     = 5
    rmax     = 25
    DEBUG=[[0.5,0,0],[0,0,0],[1,0,0],[1.5,0,0],[2,0,0],[0.5,30,0],[0.5,60,0],[0.5,90,0],[0.5,0,0.05],[0.5,0,0.1],[0.5,0,0.5],[0.5,0,2]]
    colors = ['m','b','orange','r','c','k']
    STD = np.loadtxt(output+'MCMCout/zbins_0218/param_impact/best-fit_LRG_NGC+SGC_0.dat')[5:]
    s = (STD[:,0]+STD[:,1])/2
    # plot the 2PCF multipoles   
    fontsize=14
    plt.rc('font', family='serif', size=fontsize) 
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(16,12))
    spec = gridspec.GridSpec(nrows=4,ncols=3, left = 0.06,right = 0.98,bottom=0.08,top = 0.96,height_ratios=[3, 1,3,1], hspace=0.,wspace=0.15)
    ax = np.empty((4,3), dtype=type(plt.axes))
    for zbin,param in enumerate(DEBUG):
        if zbin < 5:
            K = 0
        elif zbin < 8:
            K = 1
        else:
            K = 2    
        OBS = np.loadtxt(output+'MCMCout/zbins_0218/param_impact/best-fit_LRG_NGC+SGC_{}.dat'.format(zbin))[5:]
        for k,name in enumerate(['monopole','quadrupole']):
            #import pdb;pdb.set_trace()
            values=[np.zeros_like(STD[:,0]),STD[:,k+2]]

            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])

                if (zbin==0)|(zbin==5)|(zbin==8):
                    cind = 0
                    ax[J,K].plot(s,s**2*(STD[:,k+2]-values[j]),c='k',label='$\sigma={},V_{{vsmear}}={},V_{{ceil}}={}\%$'.format(DEBUG[0][0],DEBUG[0][1],DEBUG[0][2]))      

                if zbin !=0:
                    ax[J,K].plot(s,s**2*(OBS[:,k+2]-values[j]),c=colors[cind],alpha=0.8,label='$\sigma={},V_{{smear}}={},V_{{ceil}}={}\%$'.format(DEBUG[zbin][0],DEBUG[zbin][1],DEBUG[zbin][2])) 

                if (j==0):
                    if K==0:
                        ax[J,K].set_ylabel('$s^2 * \\xi_{}$'.format(k*2),fontsize=fontsize)

                    if k == 0:
                        if K==0:
                            plt.ylim(25,92)
                        elif K==2:
                            plt.ylim(7,80)
                        plt.legend(loc=4,prop={"size":fontsize})
                else:
                    if K==0:
                        ax[J,K].set_ylabel('$\Delta\\xi_{}$'.format(k*2),fontsize=fontsize)
                if J==3:
                    plt.xticks([5,10,15,20,25])
                    plt.xlabel('s ($h^{-1}$Mpc)',fontsize=fontsize)
                else:
                    plt.xticks([])
        cind +=1

    plt.savefig(output+'param_impact.png')
    plt.close()
elif paramtype == 'Ngal':
    print('fsat vs Ngal')
    home     = '/global/homes/j/jiaxi/SHAM/'
    datapath = '/global/cscratch1/sd/jiaxi/SHAM/'
    fileroot = '{}MCMCout/zbins_0218/fsat_vs_Ngal/'.format(home)
    gal='LOWZ'
    zmin='0.2'
    zmax='0.33'           
    z = 0.2754
    a_t = '0.78370' 
    nseed = 32
    def fsat(datadir):
        ID = np.loadtxt(datadir,usecols=3)
        return len(ID[ID!=-1])/len(ID)

    if not os.path.exists(home+'fsat_Ngal.txt'):
        f = open(home+'fsat_Ngal.txt','a')
        f.write('# zeff fsat fsat_error Ngal\n')
        for n,SHAMnum in enumerate([168500,303300,337000,370700,674000]):
            # collect the fsat of all the satellite fraction
            # for SHAM catalogues
            dataroot = datapath+'catalog/fsat/Ngal{}/'.format(SHAMnum)+'SHAM{}.dat'
            datafile = [dataroot.format(i+1) for i in range(nseed)]
            with Pool(processes = nseed) as p:
                xi1_ELG = p.map(fsat,datafile) 
            # mean and std 
            mean0 = np.mean(np.array(xi1_ELG))
            std0  = np.std(np.array(xi1_ELG))/np.sqrt(nseed)
            # for UNIT catalogues
            # write in file
            f.write('# LOWZ at {}<z<{} satellite fraction :\n'.format(zmin,zmax))
            f.write('{} {} {} {}\n'.format(z,mean0,std0,SHAMnum))
        f.close()
    else:
        mean0s = np.loadtxt(home+'fsat_Ngal.txt',usecols=(1))
        colors = ['m','c','k','r','b']
        STD = np.loadtxt(fileroot+'best-fit_LOWZ_NGC+SGC_withfsat_Ngal2.dat')[5:]
        s = (STD[:,0]+STD[:,1])/2
        # plot the 2PCF multipoles   
        fontsize=10
        plt.rc('font', family='serif', size=fontsize) 
        # plot the 2PCF multipoles   
        fig = plt.figure(figsize=(8,5))
        spec = gridspec.GridSpec(nrows=2,ncols=2, left = 0.1,right = 0.98,bottom=0.08,top = 0.96,height_ratios=[2, 1], hspace=0.,wspace=0.25)
        ax = np.empty((2,2), dtype=type(plt.axes)) 
        for n,SHAMnum in enumerate([168500,303300,337000,370700,674000]):
            # 2pcf vs SHAMnum
            Ccode = np.loadtxt('{}best-fit_{}_NGC+SGC_withfsat_Ngal{}.dat'.format(fileroot,gal,n))[5:]  
            for k,name in enumerate(['monopole','quadrupole']):
                #import pdb;pdb.set_trace()
                values=[np.zeros_like(STD[:,0]),STD[:,k+2]]
                for j in range(2):
                    ax[j,k] = fig.add_subplot(spec[j,k])
                    ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j]),c=colors[n],alpha=0.8,label='$N_{{SHAM}}$={},$f_{{sat}}$={:.1f}%'.format(SHAMnum,mean0s[n]*100))      

                    if (j==0):
                        ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2),fontsize=fontsize)
                        plt.xticks([])
                        if k == 1:
                            plt.legend(loc=1,prop={"size":fontsize})
                    else:
                        plt.xticks([5,10,15,20,25])
                        plt.xlabel('s ($h^{-1}$Mpc)',fontsize=fontsize)
                        ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2),fontsize=fontsize)
        plt.savefig(home+'fsat_vs_Ngal.png')
        plt.close()