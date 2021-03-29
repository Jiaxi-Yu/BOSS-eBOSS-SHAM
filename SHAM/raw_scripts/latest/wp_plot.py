import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib.gridspec as gridspec
from Corrfunc.theory.wp import wp
import h5py

home = '/global/homes/j/jiaxi/'
rpmin=0.01
rpmax=200
# all z range: PIP, CUTE, FCFC
for GC in ['NGC','SGC']:
    for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
        A = Table.read('{}codes/FCFC/wp_test/full/wp_log_{}_{}_rp80.dat'.format(home,gal,GC),format='ascii.no_header') # full/cut
        CUTE = Table.read('{}codes/CUTE/CUTE/wp_test/wp_CUTE_{}_{}_rp80.dat'.format(home,gal,GC),format='ascii.no_header')
        obs = Table.read('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),format='ascii.no_header')
        
        obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]
        CUTEdata = 2*np.sum(CUTE['col3'].reshape(len(obs),80),axis=1)

        fig = plt.figure(figsize=(7,8))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
        ax = np.empty((2,1), dtype=type(plt.axes))
        for j in range(2):
            ax[j,0] = fig.add_subplot(spec[j,0])
            plt.xlabel('rp (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='o',color='k')
                ax[j,0].scatter(obs['col3'],A['col4'],label='FCFC CP weight',marker='^',color='orange')
                ax[j,0].scatter(obs['col3'],CUTEdata,label='CUTE CP weight',marker='*',color='r')
                
                ax[j,0].set_ylabel('wp$(r_p)$')
                plt.legend(loc=0)
                plt.xscale('log')
                plt.ylim(0.05,700)
                plt.xlim(0.07,200)
                plt.yscale('log')
                plt.title('projected 2PCF: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,0].scatter(obs['col3'],(A['col4']-CUTEdata)/CUTEdata*100,marker='^',color='orange')
                #ax[j,0].scatter(obs['col3'],(CUTEdata-obs['col4'])/obs['col4']*100,label='CUTE CP weight',marker='*',color='r')
                plt.xscale('log')
                ax[j,0].plot(obs['col3'],np.zeros_like(np.array(obs['col3'])),label='PIP+ANG',color='k')
                ax[j,0].set_ylabel('$\Delta$wp(%)')
                ax[j,0].set_ylim(-3,3) 

        plt.savefig('{}/codes/FCFC/wp_test/wp-{}_{}_rp80.png'.format(home,gal,GC),bbox_tight=True)
        plt.close()
        
# N-body boxes: Corrfunc, FCFC_box       
dire = '/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_hlist_{}.hdf5'
# data cutting
for a in ['0.50320','0.52600','0.53780','0.54980']:
    """
    print('reading the UNIT simulation snapshot with a(t)={}'.format(a))  
    halofile = dire.format(a)  
    f=h5py.File(halofile,"r")
    sel = f["halo"]['Vpeak'][:]>300
    datac = np.zeros((len(f["halo"]['Vpeak'][:][sel]),5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = f["halo"][key][:][sel]
    f.close()   
    datasel = np.array([datac[:,2],datac[:,3],datac[:,-1],datac[:,0],datac[:,1]]).reshape(5,len(datac)).T
    np.savetxt('/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_ascii_selected/UNIT_hlist_{}_cut300.dat'.format(a),datasel)
    """
    
    # save wp
    datasel = np.loadtxt('/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_ascii_selected/UNIT_hlist_{}_cut300.dat'.format(a))
    binswp = np.unique(np.loadtxt('/global/homes/j/jiaxi/codes/FCFC/wp_test/binfile_full.dat').flatten())
    smid = 10**((np.log10(binswp[:-1])+np.log10(binswp[1:]))/2)
    wp_dat = wp(1000,80,64,binswp,datasel[:,0],datasel[:,1],datasel[:,2])
    corrfunc = np.array([binswp[:-1],binswp[1:],smid,wp_dat['wp']]).reshape(4,len(wp_dat)).T
    np.savetxt('{}codes/FCFC/wp_test/full/wp_corrfunc_UNIT_hlist_{}.dat'.format(home,a),corrfunc)
    
    # plot
    #dire = '/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_hlist_{}.hdf5'
    #for a in ['0.50320','0.52600','0.53780','0.54980']:    
    smin,smax,smid,corrfunc = np.loadtxt('{}codes/FCFC/wp_test/full/wp_corrfunc_UNIT_hlist_{}.dat'.format(home,a),unpack=True)
    A = Table.read('{}codes/FCFC/wp_test/full/wp_UNIT_hlist_{}.dat'.format(home,a),format='ascii.no_header')
    
    # plot
    fig = plt.figure(figsize=(7,8))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    for j in range(2):
        ax[j,0] = fig.add_subplot(spec[j,0])
        plt.xlabel('rp (Mpc $h^{-1}$)')
        if (j==0):
            ax[j,0].scatter(smid,corrfunc,label='Corrfunc.theory.wp',marker='o',color='k')
            ax[j,0].scatter(smid,A['col4'],label='FCFC_box',marker='*',color='r')  
            ax[j,0].set_ylabel('wp$(r_p)$')
            plt.legend(loc=0)
            plt.xscale('log')
            plt.ylim(0.1,2e3)
            plt.xlim(0.07,200)
            plt.yscale('log')
            plt.title('projected 2PCF in truncated UNIT with a(t)={}'.format(a))
        if (j==1):
            ax[j,0].scatter(smid,(A['col4']-corrfunc)/corrfunc*100,marker='*',color='r')
            ax[j,0].plot(smid,np.zeros_like(corrfunc),color='k')
            ax[j,0].set_ylabel('$\Delta$wp(%)')
            ax[j,0].set_ylim(-3,3) 
            plt.xscale('log')

    plt.savefig('{}/codes/FCFC/wp_test/wp-UNIT_{}.png'.format(home,a),bbox_tight=True)
    plt.close()
    
"""  
# zbins
for GC in ['NGC']:
    for gal,ver,zrange in zip(['LRG','ELG'],['v7_2','v7'],['z0.6z0.7','z0.9z1.1']):
        A = Table.read('{}codes/FCFC/wp_test/cut/wp_log_{}_{}_rp80_{}.dat'.format(home,gal,GC,zrange),format='ascii.no_header')
        obs = Table.read('{}catalog/nersc_zbins_wp_mps_{}/wp_log_{}_NGC+SGC_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,gal,ver,zrange[1:4],zrange[5:]),format='ascii.no_header')
        obs = obs[(obs['col3']>=rpmin)&(obs['col3']<rpmax)]

        fig = plt.figure(figsize=(7,8))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
        ax = np.empty((2,1), dtype=type(plt.axes))
        for j in range(2):
            ax[j,0] = fig.add_subplot(spec[j,0])
            plt.xlabel('rp (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,0].scatter(obs['col3'],A['col4'],label='FCFC CP weight',marker='^',color='orange')
                ax[j,0].scatter(obs['col3'],obs['col4'],label='PIP+ANG',marker='*',color='k')
                ax[j,0].set_ylabel('wp$(r_p)$')
                plt.legend(loc=0)
                plt.xscale('log')
                plt.ylim(1,700)
                plt.yscale('log')
                plt.title('projected 2PCF: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,0].scatter(obs['col3'],(A['col4']-obs['col4'])/obs['col4']*100,label='data CP',marker='^',color='orange')
                plt.xscale('log')
                ax[j,0].scatter(obs['col3'],np.ones_like(np.array(obs['col3'])),label='PIP+ANG',marker='*',color='k')
                ax[j,0].set_ylabel('$\Delta$wp(%)')

        plt.savefig('{}_wp-{}_{}_rp80.png'.format(zrange,gal,GC),bbox_tight=True)
        plt.close()

"""  