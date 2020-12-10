import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool 
from itertools import repeat
import sys

# variables
gal     = 'LRG'
GC      = 'SGC'
nseed   = 20
home    = '/global/cscratch1/sd/jiaxi/SHAM/'
mode    = 'close_chi2'
func    = 'bispec'
label   = int(sys.argv[1])

# random seeds
seed = [x for x in range(nseed)]
seed1 = [x+30 for x in range(nseed)]

def sham_tpcf(seed,seed1,sigM,sigV,Mtrun):      
    x00    = sham_cal(seed,sigM,sigV,Mtrun)
    x01    = sham_cal(seed1,sigM,sigV,Mtrun)
    return [x00,x01]

def sham_cal(seednum,sigma_high,sigma,v_high):
    bispec = np.loadtxt('outputs/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,sigma_high,int(sigma),int(10**v_high),seednum))
    return bispec[:,-2]

if label:
    pars = [[0.14864245,131.20083923,5.24729866],[0.59092776,109.82603879,4.87587909]]
    for param in pars:
        for i,seednum in enumerate(seed):
            os.system('./bispec -i testdata/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat -o outputs/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,param[0],int(param[1]),int(10**param[2]),seednum,gal,GC,param[0],int(param[1]),int(10**param[2]),seednum))

        for i,seednum in enumerate(seed1):
            os.system('./bispec -i testdata/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat -o outputs/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,param[0],int(param[1]),int(10**param[2]),seednum,gal,GC,param[0],int(param[1]),int(10**param[2]),seednum))

else:
    # calculate the SHAM 2PCF
    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf,list(zip(seed,seed1,repeat(np.float32(0.14864245)),repeat(131.20083923),repeat(np.float32(5.24729866))))) 

    with Pool(processes = nseed) as p:
        xi0_ELG = p.starmap(sham_tpcf,list(zip(seed,seed1,repeat(np.float32(0.59092776)),repeat(109.82603879),repeat(np.float32(4.87587909))))) 
    s = np.loadtxt('outputs/SHAM_LRG_SGC_sigma0.591_Vsmear109_Vceil75141_seed49.dat')[:,0]
    # plot the mean
    fig = plt.figure(figsize=(5,6))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,1), dtype=type(plt.axes))
    k=0
    values=[np.zeros_like(s),np.mean(np.array(xi1_ELG).reshape(40,20),axis=0)]
    for j in range(2):
        ax[j,k] = fig.add_subplot(spec[j,k])
        plt.xlabel('$\\theta$')

        #plt.xscale('log')
        if (j==0):
            ax[j,k].plot(s,(np.mean(np.array(xi1_ELG).reshape(40,20),axis=0)),color='c',label='$\chi^2=75.39$')        
            ax[j,k].plot(s,(np.mean(np.array(xi0_ELG).reshape(40,20),axis=0)),c='m',label='$\chi^2=75.33$')
            ax[j,k].set_ylabel('bispec')
            plt.legend(loc=0)
            plt.yscale('log')
            plt.title('bispectrum: {} in {}'.format(gal,GC))
            plt.ylim(5e8,3e9)
        if (j==1):
            ax[j,k].set_ylabel('$\Delta$ bispec(%)')
            ax[j,k].plot(s,np.zeros_like(s),color='c',label='$\chi^2=75.39$')        
            ax[j,k].plot(s,(np.mean(np.array(xi0_ELG).reshape(40,20),axis=0)-np.mean(np.array(xi1_ELG).reshape(40,20),axis=0))*100/np.mean(np.array(xi1_ELG).reshape(40,20),axis=0),c='m',label='$\chi^2=75.33$')
    plt.savefig('{}_{}_{}_{}.png'.format(func,mode,gal,GC),bbox_tight=True)
    plt.close()
    
    data = [np.array(xi1_ELG).reshape(40,20),np.array(xi0_ELG).reshape(40,20)]
    color = ['m','c']
    name = ['chi2_75.33','chi2_75.39']
    for q,labels in enumerate(['$\chi^2=75.33$','$\chi^2=75.39$']):
        fig = plt.figure(figsize=(5,6))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,1), dtype=type(plt.axes))
        k=0
        values=[np.zeros_like(s),np.mean(data[q],axis=0)]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            for i in range(2*nseed):
                ax[j,k].plot(s,data[q][i],color=color[q],lw=0.5)

            plt.xlabel('$\\theta$')
            #plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('bispectrum')
                plt.title('bispectrum: {} in {}'.format(gal,GC))
                plt.yscale('log')
                ax[j,k].plot(s,(np.mean(data[q],axis=0)-values[j]),c='k',label=labels)
                plt.legend(loc=0)
                plt.ylim(1e8,3e9)
            if (j==1):
                ax[j,k].set_ylabel('$\Delta$ bispec')
        plt.savefig('{}_{}_{}_{}_{}.png'.format(func,mode,gal,GC,name[q]),bbox_tight=True)
        plt.close()

"""
# out of memory
def sham_cal(seednum,sigma_high,sigma,v_high):
    if label:
        os.system('./bispec -i testdata/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat -o outputs/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,sigma_high,int(sigma),int(10**v_high),seednum,gal,GC,sigma_high,int(sigma),int(10**v_high),seednum))
        return 0
    else:
        bispec = np.loadtxt('outputs/SHAM_{}_{}_sigma{:.3f}_Vsmear{}_Vceil{}_seed{}.dat'.format(gal,GC,sigma_high,int(sigma),int(10**v_high),seednum))
        return bispec
"""