import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import matplotlib.gridspec as gridspec
from Corrfunc.theory.wp import wp
import sys
import re

home = '/global/homes/j/jiaxi/'
rpmin=0.01
rpmax=200
if sys.argv[1] == 'lightcone':
    # all z range: PIP, CUTE, FCFC
    for GC in ['NGC','SGC']:
        for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
            A = Table.read('{}codes/FCFC/wp_test/full/wp_log_{}_{}_rp80.dat'.format(home,gal,GC),format='ascii.no_header') # full/cut
            CUTE = Table.read('{}codes/CUTE/CUTE/wp_test/wp_CUTE_{}_{}_rp80.dat'.format(home,gal,GC),format='ascii.no_header')
            obs = Table.read('{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver),format='ascii.no_header')

            sel = (obs['col3']>=rpmin)&(obs['col3']<rpmax)
            obs = obs[sel]
            CUTEdata = 2*np.sum(CUTE['col3'].reshape(33,80),axis=1)[sel]
            A = A[sel]

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
                    plt.ylim(1e-2,1e3)
                    plt.yscale('log')
                    plt.xlim(0.07,200)
                    plt.xscale('log')
                    plt.title('projected 2PCF: {} in {}'.format(gal,GC))
                if (j==1):
                    ax[j,0].plot(obs['col3'],np.zeros_like(np.array(obs['col3'])),color='r')
                    ax[j,0].scatter(obs['col3'],(A['col4']-CUTEdata)/CUTEdata*100,marker='^',color='orange')
                    #ax[j,0].scatter(obs['col3'],(CUTEdata-obs['col4'])/obs['col4']*100,label='CUTE CP weight',marker='*',color='r')

                    plt.xlim(0.07,200)
                    plt.xscale('log')
                    ax[j,0].set_ylabel('$\Delta$wp(%)')
                    ax[j,0].set_ylim(-3,3) 

            plt.savefig('{}/codes/FCFC/wp_test/wp-{}_{}_rp80.png'.format(home,gal,GC))
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

            plt.savefig('{}_wp-{}_{}_rp80.png'.format(zrange,gal,GC))
            plt.close()

    """  
elif sys.argv[1]=='box':
    # N-body boxes: Corrfunc, FCFC_box       
    dire = '/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_hlist_{}.hdf5'
    for a in ['0.50320','0.52600','0.53780','0.54980']:
        """
        #import h5py
        # truncate UNIT into smaller catalogues
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
        """
        # save wp
        datasel = np.loadtxt('/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_ascii_selected/UNIT_hlist_{}_cut300.dat'.format(a))
        binswp = np.unique(np.loadtxt('/global/homes/j/jiaxi/codes/FCFC/wp_test/binfile_full.dat').flatten())
        smid = 10**((np.log10(binswp[:-1])+np.log10(binswp[1:]))/2)
        wp_dat = wp(1000,70,64,binswp,datasel[:,0],datasel[:,1],datasel[:,2])
        corrfunc = np.array([binswp[:-1],binswp[1:],smid,wp_dat['wp']]).reshape(4,len(wp_dat)).T
        np.savetxt('{}codes/FCFC/wp_test/full_logpi/wp_corrfunc_UNIT_hlist_{}.dat'.format(home,a),corrfunc)
        """
        # plot
        #dire = '/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_hlist_{}.hdf5'
        #for a in ['0.50320','0.52600','0.53780','0.54980']:    
        smin,smax,smid,corrfunc = np.loadtxt('{}codes/FCFC/wp_test/full_logpi/wp_corrfunc_UNIT_hlist_{}.dat'.format(home,a),unpack=True)
        A = Table.read('{}codes/FCFC/wp_test/full_logpi/wp_UNIT_hlist_{}.dat'.format(home,a),format='ascii.no_header')
        Aa = Table.read('{}codes/FCFC/wp_test/full_logpi/wp_UNIT_hlist_{}.no0.dat'.format(home,a),format='ascii.no_header')

        # plot
        fig = plt.figure(figsize=(7,8))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
        ax = np.empty((2,1), dtype=type(plt.axes))
        for j in range(2):
            ax[j,0] = fig.add_subplot(spec[j,0])
            plt.xlabel('rp (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,0].scatter(smid,corrfunc,label='Corrfunc.theory.wp',marker='o',color='k')
                ax[j,0].scatter(smid,A['col4'],label='FCFC_box_pi-with0',marker='*',color='r')  
                ax[j,0].scatter(smid,Aa['col4'],label='FCFC_box_pi-no0',marker='X',color='b')  
                ax[j,0].set_ylabel('wp$(r_p)$')
                plt.legend(loc=0)
                plt.ylim(0.01,2e3)
                plt.xlim(0.07,200)
                plt.xscale('log')
                plt.yscale('log')
                plt.title('wp in truncated UNIT with a(t)={}'.format(a))
            if (j==1):
                ax[j,0].scatter(smid,(A['col4']-corrfunc)/corrfunc*100,marker='*',color='r')
                ax[j,0].scatter(smid,(Aa['col4']-corrfunc)/corrfunc*100,marker='X',color='b')
                ax[j,0].plot(smid,np.zeros_like(corrfunc),color='k')
                ax[j,0].set_ylabel('$\Delta$wp(%)')
                plt.xlim(0.07,200)
                plt.xscale('log')
                ax[j,0].set_ylim(-5,5) 

        plt.savefig('{}/codes/FCFC/wp_test/wp-UNIT_{}.wo0.png'.format(home,a))
        plt.close()
    
elif sys.argv[1]=='boxlogpi':
    dire = '/global/cscratch1/sd/jiaxi/SHAM/catalog/UNIT_hlist_{}.hdf5'
    for a in ['0.50320','0.52600','0.53780','0.54980']:   
        # plot  
        A = Table.read('{}codes/FCFC/wp_test/full_logpi/wp_UNIT_hlist_{}.dat'.format(home,a),format='ascii.no_header')
        L = Table.read('{}codes/FCFC/wp_test/full_logpi/wp_UNIT_hlist_{}.70.dat'.format(home,a),format='ascii.no_header')

        # plot
        fig = plt.figure(figsize=(7,8))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
        ax = np.empty((2,1), dtype=type(plt.axes))
        for j in range(2):
            ax[j,0] = fig.add_subplot(spec[j,0])
            plt.xlabel('rp (Mpc $h^{-1}$)')
            if (j==0):
                ax[j,0].scatter(A['col1'],L['col4'],label='pi_linear',marker='o',color='k')
                ax[j,0].scatter(A['col1'],A['col4'],label='pi_log',marker='*',color='r')  
                ax[j,0].set_ylabel('wp$(r_p)$')
                plt.legend(loc=0)
                plt.ylim(0.01,2e3)
                plt.xlim(0.07,200)
                plt.xscale('log')
                plt.yscale('log')
                plt.title('wp in truncated UNIT with a(t)={}'.format(a))
            if (j==1):
                ax[j,0].scatter(A['col1'],(A['col4']-L['col4'])/L['col4']*100,marker='*',color='r')
                ax[j,0].plot(A['col1'],np.zeros_like(L['col4']),color='k')
                ax[j,0].set_ylabel('$\Delta$wp(%)')
                plt.xlim(0.07,200)
                plt.xscale('log')
                ax[j,0].set_ylim(-0.2,0.2) 


        plt.savefig('{}/codes/FCFC/wp_test/wp-UNIT_{}.png'.format(home,a))
        plt.close()   
        
elif sys.argv[1]=='2D':
    for GC in ['NGC','SGC']:
        for gal,ver in zip(['LRG','ELG'],['v7_2','v7']):
            CUTE = Table.read('{}codes/CUTE/CUTE/wp_test/wp_CUTE_{}_{}_rp80.dat'.format(home,gal,GC),format='ascii.no_header')
            X, Y = np.meshgrid(np.unique(CUTE['col2']),np.unique(CUTE['col1']))
            fig = plt.figure(figsize=(21,7))
            spec = gridspec.GridSpec(nrows=1,ncols=3)#, height_ratios=[4, 1])#, hspace=0.3)
            ax = np.empty((1,3), dtype=type(plt.axes))
            j=0
            for ic,pn in enumerate(['dd','dr','rr']):
                filename = '{}codes/FCFC/wp_test/full/log_{}_{}_rp80.{}'.format(home,gal,GC,pn)
                A = Table.read(filename,format='ascii.no_header')['col5'] # full/cut
                columns = []
                with open(filename, 'r') as td:
                    for line in td:
                        if line[0] == '#':
                            info = re.split(' +', line)
                            columns.append(info)
                norm = float(columns[2][-1][:-1])
                ax[j,ic] = fig.add_subplot(spec[j,ic])
                plt.xlabel('rp (Mpc $h^{-1}$)')    
                if ic != 2:
                    cute = CUTE['col{}'.format(4+ic)].reshape(33,80).T.flatten()
                    c=ax[j,ic].pcolormesh(X,Y,((A*norm-cute)/cute).reshape(80,33)*100,cmap='RdBu', vmin=-1, vmax=1)
                else:
                    cute = CUTE['col{}'.format(5+ic)].reshape(33,80).T.flatten()
                    c = ax[j,ic].pcolormesh(X,Y,((A*norm-cute)/cute).reshape(80,33)*100,cmap='RdBu', vmin=-1, vmax=1)
                ax[j,ic].set_ylabel('$\pi$')
                cbar = fig.colorbar(c,ax=ax[j,ic])
                cbar.ax.set_ylabel('(FCFC/CUTE-1)*100', rotation=270)
                #plt.legend(loc=0)
                plt.ylim(0,80)
                #plt.yscale('log')
                plt.xlim(0.07,200)
                plt.xscale('log')
                plt.title('projected 2PCF {} counts: {} in {}'.format(pn,gal,GC))

            plt.savefig('{}/codes/FCFC/wp_test/full/pairs-{}_{}_rp80.png'.format(home,gal,GC))
            plt.close()
