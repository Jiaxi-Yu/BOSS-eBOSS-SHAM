import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack
from astropy.table import Table
import sys
import matplotlib.gridspec as gridspec 
import astropy.io.fits as fits
from  glob import glob 

task     = sys.argv[1]
# together, NGCSGC, separate scripts
gal      = 'LOWZ'
GC       = 'NGC+SGC'
rscale   = 'linear' # 'log'
function = 'mps'
zmins = ['0.2', '0.33','0.2']
zmaxs = ['0.33','0.43','0.43']
rmin     = 5
rmax = 25
multipole= 'quad' # 'mono','quad','hexa'
pre = '/'
home     = '/home/astro/jiayu/Desktop/SHAM/'
fontsize=15
plt.rc('font', family='serif', size=fontsize)

if task == 'together':
    mpsnum = 1
    fig = plt.figure(figsize=(12,5*mpsnum))
    spec = gridspec.GridSpec(nrows=2*mpsnum,ncols=3, left = 0.09,right = 0.96,bottom=0.15,top = 0.9, hspace=0,wspace=0.05,height_ratios=[3, 1])#,height_ratios=[3, 1,3,1]
    ax = np.empty((2*mpsnum,3), dtype=type(plt.axes))
    
    for zmin,zmax,K in zip(zmins,zmaxs,range(len(zmins))):
        fileroot = '{}MCMCout/zbins_0218/{}{}_{}_{}_{}_z{}z{}/best-fit_{}_{}.dat'.format(home,pre,function,rscale,gal,'NGC+SGC',zmin,zmax,gal,'NGC+SGC')
        if (rscale=='linear')&(function=='mps'):
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
            # generate s bins
            bins  = np.arange(rmin,rmax+1,1)
            nbins = len(bins)-1
            binmin = rmin
            binmax = rmax
            s = (bins[:-1]+bins[1:])/2

            # covariance matrices and observations
            if (gal == 'LRG')|(gal=='ELG'):
                obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
                covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
            else:
                obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_{}_DR12v5_z{}z{}.mps'.format(home,gal,GC,zmin,zmax)
                covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
            
            # Read the covariance matrices and observations
            hdu = fits.open(covfits) #
            mock = hdu[1].data[GC+'mocks']
            Nmock = mock.shape[1] 
            hdu.close()
            if (gal == 'LRG')|(gal=='ELG'):
                Nstot=200
            else:
                Nstot=100
            mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
            covcut  = cov(mocks).astype('float32')
            obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
            if gal == 'ELG':
                OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
            else:
                OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
            covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
            print('the covariance matrix and the observation 2pcf vector are ready.')
        else:
            print('wrong 2pcf function input')

        if rscale == 'linear':
            Ccode = np.loadtxt(fileroot)[binmin:binmax]
            Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[binmin:binmax]
        else:
            Ccode = np.loadtxt(fileroot)[1:]
            Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[1:]
        disp = np.std(mocks,axis=1)

        #for name,k in zip(['monopole','quadrupole'],range(2)):
        k=1
        values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
        err   = [np.ones(nbins),s**2*disp[k*nbins:(k+1)*nbins]]
        for j in range(2):
            J = j#2*k+j
            ax[J,K] = fig.add_subplot(spec[J,K])
            ax[J,K].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='r',alpha=0.7,label=r'$v_{\rm smear}$')
            ax[J,K].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='k',label='$\sigma_{\Delta v}$')
            ax[J,K].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],\
                    color='r', marker='o',ecolor='r',ls="none",\
                    label='obs')
            

            if rscale=='log':
                plt.xscale('log')
            
            if (j==0):
                plt.title('${}<z<{}$'.format(zmin,zmax))
                if K==0:
                    ax[J,K].set_ylabel(r'$s^2 \xi_{}\,(h^{{-2}}\,Mpc^2)$'.format(k*2),fontsize=fontsize)
                else:
                    plt.yticks(alpha=0)#95,100,105,110,115,120,125,130])

                plt.xticks([])
                if k == 0:
                    plt.ylim(91,130)
                else:
                    plt.ylim(-84,37)
                    plt.legend(loc=1)
            else:
                if K==0:
                    if k==0:
                        ax[J,K].set_ylabel(r'$\Delta\xi_0/\epsilon_{{obs,\xi_0}}$',fontsize=fontsize)
                    else:
                        ax[J,K].set_ylabel(r'$\Delta\xi_2/\epsilon_{{obs,\xi_2}}$',fontsize=fontsize)
                else:
                    plt.yticks(alpha=0)
                plt.ylim(-4,1)
                plt.yticks([-3,0])
                plt.xlabel(r'$s\,(h^{-1}\,Mpc)$',fontsize=fontsize)
                plt.xticks([5,10,15,20,25])

    plt.savefig(home+'LOWZ_xi2_vmsear-vs-std.pdf')
    plt.close()
elif task =='NGCSGC':
    fig = plt.figure(figsize=(12,8))
    spec = gridspec.GridSpec(nrows=4,ncols=3, left = 0.09,right = 0.96,bottom=0.08,top = 0.95,height_ratios=[3, 1,3,1], hspace=0,wspace=0.05)
    ax = np.empty((4,3), dtype=type(plt.axes))
    colours = ['m','b']
    
    for zmin,zmax,K in zip(zmins,zmaxs,range(len(zmins))):
        for kk,GC in enumerate(['NGC','SGC']):
            fileroot = '{}MCMCout/zbins_0218/{}{}_{}_{}_{}_z{}z{}/best-fit_{}_{}.dat'.format(home,pre,function,rscale,gal,GC,zmin,zmax,gal,GC)
            if (rscale=='linear')&(function=='mps'):
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
                # generate s bins
                bins  = np.arange(rmin,rmax+1,1)
                nbins = len(bins)-1
                binmin = rmin
                binmax = rmax
                s = (bins[:-1]+bins[1:])/2

                # covariance matrices and observations
                if (gal == 'LRG')|(gal=='ELG'):
                    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
                    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
                else:
                    obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_{}_DR12v5_z{}z{}.mps'.format(home,gal,GC,zmin,zmax)
                    covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
                
                # Read the covariance matrices and observations
                hdu = fits.open(covfits) #
                mock = hdu[1].data[GC+'mocks']
                Nmock = mock.shape[1] 
                hdu.close()
                if (gal == 'LRG')|(gal=='ELG'):
                    Nstot=200
                else:
                    Nstot=100
                mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
                covcut  = cov(mocks).astype('float32')
                obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
                if gal == 'ELG':
                    OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
                else:
                    OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
                covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
                print('the covariance matrix and the observation 2pcf vector are ready.')
            else:
                print('wrong 2pcf function input')


            if rscale == 'linear':
                Ccode = np.loadtxt(fileroot)[binmin:binmax]
                Ccode1 = np.loadtxt(fileroot[:-4]+'_deltav.dat')[binmin:binmax]
            else:
                Ccode = np.loadtxt(fileroot)[1:]
                Ccode1 = np.loadtxt(fileroot[:-4]+'_deltav.dat')[1:]
            disp = np.std(mocks,axis=1)

            k=1
            values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
            err   = [np.ones(nbins),s**2*disp[k*nbins:(k+1)*nbins]]

            for j in range(2):
                J = 2*kk+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                ax[J,K].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c=colours[kk],alpha=0.7,label=r'$v_{{\rm smear}}$ in {}'.format(GC))
                ax[J,K].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='k',label='$\sigma_{{\Delta v}}$ in {}'.format(GC))
                ax[J,K].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],\
                        color=colours[kk], marker='o',ecolor=colours[kk],ls="none",\
                        label='obs in {}'.format(GC))
                if (K == 0)&(j==0):
                    plt.legend(loc=1)

                if rscale=='log':
                    plt.xscale('log')
                
                if J==0:
                    plt.title('${}<z<{}$'.format(zmin,zmax))

                if (j==0):
                    if K==0:
                        ax[J,K].set_ylabel(r'$s^2 \xi_{}\,(h^{{-2}}\,Mpc^2)$'.format(k*2),fontsize=fontsize)
                    else:
                        plt.yticks([])#95,100,105,110,115,120,125,130])

                    plt.xticks([])
                    plt.ylim(-84,37)
                else:
                    if K==0:
                        if k==0:
                            ax[J,K].set_ylabel(r'$\Delta\xi_0/\epsilon_{{obs,\xi_0}}$',fontsize=fontsize)
                        else:
                            ax[J,K].set_ylabel(r'$\Delta\xi_2/\epsilon_{{obs,\xi_2}}$',fontsize=fontsize)
                    else:
                        plt.yticks(alpha=0)
                    plt.ylim(-4,1.5)
                    plt.yticks([-3,0])

                if J==3:
                    plt.xlabel(r'$s\,(h^{-1}\,Mpc)$',fontsize=fontsize)
                    plt.xticks([5,10,15,20,25])
                else:
                    plt.xticks([])
                

    plt.savefig(home+'LOWZ_xi2_vmsear-vs-std_NGCSGC.pdf')
    plt.close()
elif task == 'separate':
    fileroot = '{}MCMCout/zbins_0218/{}{}_{}_{}_{}_z{}z{}/best-fit_{}_{}.dat'.format(home,pre,function,rscale,gal,'NGC+SGC',zmin,zmax,gal,'NGC+SGC')
    if (rscale=='linear')&(function=='mps'):
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
        # generate s bins
        bins  = np.arange(rmin,rmax+1,1)
        nbins = len(bins)-1
        binmin = rmin
        binmax = rmax
        s = (bins[:-1]+bins[1:])/2

        # covariance matrices and observations
        if (gal == 'LRG')|(gal=='ELG'):
            obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
            covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)
        else:
            obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_{}_DR12v5_z{}z{}.mps'.format(home,gal,GC,zmin,zmax)
            covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
        
        # Read the covariance matrices and observations
        hdu = fits.open(covfits) #
        mock = hdu[1].data[GC+'mocks']
        Nmock = mock.shape[1] 
        hdu.close()
        if (gal == 'LRG')|(gal=='ELG'):
            Nstot=200
        else:
            Nstot=100
        mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
        covcut  = cov(mocks).astype('float32')
        obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
        if gal == 'ELG':
            OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
        else:
            OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
        covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
        print('the covariance matrix and the observation 2pcf vector are ready.')
    else:
        print('wrong 2pcf function input')

    if rscale == 'linear':
        Ccode = np.loadtxt(fileroot)[binmin:binmax]
        Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[binmin:binmax]
    else:
        Ccode = np.loadtxt(fileroot)[1:]
        Ccode1 = np.loadtxt(fileroot[:-4]+'-deltav.dat')[1:]
    disp = np.std(mocks,axis=1)

    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.2,wspace=0.2)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for name,k in zip(['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
        err   = [np.ones(nbins),s**2*disp[k*nbins:(k+1)*nbins]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            # mocks mean and std
            ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='c',label=r'SHAM {} $v_{{\rm smear}}$'.format(GC))
            ax[j,k].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='m',label='SHAM {} $\Delta'.format(GC))
            ax[j,k].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs {}'.format(GC))

            plt.xlabel('s ($h^{-1}$Mpc)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                if k==0:
                    pass
                    #plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('correlation function {} in {} at {}<z<{}'.format(name,GC,zmin,zmax))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2))
                plt.ylim(-3,3)

    plt.savefig('{}_{}_z{}z{}_s{}-{}Mpch-1-quadtest.png'.format(gal,GC,zmin,zmax,rmin,rmax),bbox_tight=True)
    plt.close()

    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.2,wspace=0.2)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for name,k in zip(['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),OBS[k*len(s):(k+1)*len(s)]]        
        err   = [np.ones(nbins),np.ones(nbins)]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            # mocks mean and std
            ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='c',label=r'SHAM {} $v_{{\rm smear}}$'.format(GC))
            ax[j,k].plot(s,s**2*(Ccode1[:,k+2]-values[j])/err[j],c='m',label='SHAM {} $\Delta v$'.format(GC))
            ax[j,k].errorbar(s,s**2*(OBS[k*len(s):(k+1)*len(s)]-values[j])/err[j],s**2*disp[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs {}'.format(GC))

            plt.xlabel('s ($h^{-1}$Mpc)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                if k==0:
                    pass
                    #plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('correlation function {} in {} at {}<z<{}'.format(name,GC,zmin,zmax))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$'.format(k*2))
                plt.ylim(-15,5)

    plt.savefig('{}_{}_z{}z{}_s{}-{}Mpch-1-quadtest-absdiff.png'.format(gal,GC,zmin,zmax,rmin,rmax),bbox_tight=True)
    plt.close()
elif task == 'scripts':
    import pymultinest
    for zmin,zmax,K in zip(zmins,zmaxs,range(len(zmins))):
        if (rscale=='linear')&(function=='mps'):
            if (zmin=='0.2')&(zmax=='0.33'):            
                SHAMnum = 337000
                z = 0.2754
                a_t = '0.78370' 
                dvnorm = [20.9,21.5]#[20.9,21.5]
            elif zmin=='0.33':
                SHAMnum = 258000
                z = 0.3865
                a_t = '0.71730'
                dvnorm = [27.5,28.2]#[27.5,28.2]
            elif (zmin=='0.2')&(zmax=='0.43'): 
                SHAMnum = 295000
                z = 0.3441
                a_t = '0.74980'
                dvnorm = [23.8,24.3]#[23.8,24.3]
        fileroot = '{}MCMCout/zbins_0218/{}{}_{}_{}_{}_z{}z{}/multinest_'.format(home,pre,function,rscale,gal,GC,zmin,zmax,gal,GC)        
        MCdir='/home/astro/jiayu/Desktop/SHAM/MCMCout/zbins_0218/mps_linear_LRG_NGC+SGC_z{}z{}'.format(zmin,zmax)
        a = pymultinest.Analyzer(3, outputfiles_basename = fileroot)
        print(a.get_best_fit()['parameters'],(dvnorm[0]+dvnorm[1])/2)