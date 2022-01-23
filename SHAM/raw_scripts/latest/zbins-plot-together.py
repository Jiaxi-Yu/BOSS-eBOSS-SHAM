import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
import os
import warnings
import matplotlib.pyplot as plt
from glob import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import pymultinest
import corner
import h5py

# variables
gal      = sys.argv[1]
date     = '0218'
task     = sys.argv[2] # 2pcf, wp,posteriors
# 2pcf, wp
pre      = '/'
pimax    = 30
GC       = 'NGC+SGC'
if gal == 'LRG':
    rscale   = 'log'
else:
    rscale = 'linear'
function = 'mps' 
nseed    = 30
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
cols = ['col4','col5']
# read the posterior file
if date == '0729':
    parameters = ["sigma","Vsmear"]
    if pre[:7] == 'mocks10':
        rmin = 15
        rmax = 35
    elif rscale == 'linear':
        rmin = 5
        rmax = 25
    else:
        rmin = 5
        rmax = 30
        cols = ['col4','col5']
else:
    parameters = ["sigma","Vsmear","Vceil"]
    if date == '0726':
        rmin = 12
        rmax = 40
    else:
        rmin     = 5
        if rscale =='linear':
            rmax = 25
        else:
            rmax = 30
            cols = ['col4','col5']
            
nthread  = 1
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=30
home     = '/home/astro/jiayu/Desktop/SHAM/'
if gal == 'LRG':
    zmins = [0.6,0.65,0.6,0.7,0.8,0.6] #0.6
    zmaxs = [0.7,0.8, 0.8,0.9,1.0,1.0] #0.8
    ver = 'v7_2'
elif gal == 'ELG':
    zmins = [0.6,0.7,0.8,0.9]
    zmaxs = [0.8,0.9,1.0,1.1]
    ver = 'v7'
elif gal == 'CMASS':
    zmins = [0.43,0.51,0.57,0.43]
    zmaxs = [0.51,0.57,0.7,0.7]
elif gal == 'LOWZ':
    zmins = [0.2, 0.33,0.2]
    zmaxs = [0.33,0.43,0.43]
zbinnum = len(zmins)
colors = ['m','b','orange','r','c','k']
#cols = ['col4','col5']
ver1 = 'v7_2'

samples = [x for x in range(zbinnum)]
bestfits = [x for x in range(zbinnum)]
Ccodes = [x for x in range(zbinnum)]
errbarshams = [x for x in range(zbinnum)]
obscfs= [x for x in range(zbinnum)]
errbars= [x for x in range(zbinnum)]
OBSwps= [x for x in range(zbinnum)]
wps= [x for x in range(zbinnum)]
errbarwps= [x for x in range(zbinnum)]
OBSwppis= [x for x in range(zbinnum)]
wppis= [x for x in range(zbinnum)]
errbarwppis= [x for x in range(zbinnum)]
pdfs= [x for x in range(zbinnum)]

########################################################
for zbin in range(zbinnum):
    zmin,zmax = zmins[zbin],zmaxs[zbin]
    if (gal == 'LRG')&(zbin==zbinnum-1):
        rscale ='linear'

    # getdist results
    if (gal == 'CMASS')&(zmax == 0.7)&(date =='0729'):
        pre = '0218_'
    else:
        pre = '/'
    fileroot = '{}MCMCout/zbins_{}/{}{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,pre,function,rscale,gal,GC,zmin,zmax)
    if (date == '0729'):
        parameters = ["sigma","Vsmear"]
    else:
        parameters = ["sigma","Vsmear","Vceil"]
    npar = len(parameters)
    a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
    sample = loadMCSamples(fileroot)
    
    # read s bins
    if (gal == 'LRG')|(gal=='ELG'):
        if pre[:4]=='mock':
            binfile = Table.read(home+'binfile_fine.dat',format='ascii.no_header')
            covfits = home+'catalog/wp_diff-pi/nosys_FKP/EZmocks_nosys_2PCF_{}_z{}z{}_quad.fits.gz'.format(rscale,zmin,zmax)
            obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.mocks'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
        else:
            binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header')
            covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
            obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
        sel = (binfile['col3']<rmax)&(binfile['col3']>=rmin)
        bins  = np.unique(np.append(binfile['col1'][sel],binfile['col2'][sel]))
        s = binfile['col3'][sel]
        nbins = len(bins)-1
        binmin = np.where(binfile['col3']>=rmin)[0][0]
        binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    else:
        # generate s bins
        bins  = np.arange(rmin,rmax+1,1)
        nbins = len(bins)-1
        binmin = rmin
        binmax = rmax
        s = (bins[:-1]+bins[1:])/2
        # filenames
        if pre[:4]=='mock':
            obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.15-35.mocks'.format(home,gal,zmin,zmax)
        else:
            obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.mps'.format(home,gal,zmin,zmax)
        covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
        obstool = ''

    if (gal == 'LRG')&(zbin==zbinnum-1):
        bins  = np.arange(5,26,1)
        binmin = 5
        binmax = 25
        rmax=25
        obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(home,gal,ver,function,rscale,gal,GC)
        covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(home,gal,ver,function,rscale,gal,multipole)


    # Read the covariance matrices 
    hdu = fits.open(covfits) # cov([mono,quadru])
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1]
    hdu.close()
    Nstot = int(mock.shape[0]/2)

    # observations
    if pre[:4]=='mock':
        if gal == 'LRG':
            obscf = Table.read(obs2pcf,format='ascii.no_header')[1:]
        else:
            obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]
    else:  
        obscf = Table.read(obs2pcf,format='ascii.no_header')
        if rscale == 'linear':
            obscf= obscf[(obscf['col1']<rmax)&(obscf['col1']>=rmin)]
        else:
            obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    # prepare OBS, covariance and errobar for chi2
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
    covcut  = cov(mocks).astype('float32')
    OBS   = append(obscf[cols[0]],obscf[cols[1]]).astype('float32')# LRG columns are s**2*xi
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
    if pre[:5]=='mocks':
        errbar = np.std(mocks,axis=1)/np.sqrt(float(pre[5:7]))
    else:
        errbar = np.std(mocks,axis=1)        

    # plot wp with errorbars
    wp = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
    if (gal == 'LRG')|(gal=='ELG'):
        if rscale == 'linear':
            covfitswp  = '{}catalog/nersc_{}_{}_{}/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,'wp',gal,ver,gal,zmin,zmax) 
            obs2pcfwp  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(home,gal,ver,gal,GC,ver)
        elif rscale == 'log':
            covfitswp = '{}catalog/nersc_zbins_wp_mps_{}/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,gal,gal,zmin,zmax) 
            obs2pcfwp  = '{}catalog/nersc_zbins_wp_mps_{}/wp_log_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,gal,GC,ver1,zmin,zmax)
        colwp   = 'col3'
    else:
        obs2pcfwp = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}.wp'.format(home,gal,zmin,zmax)
        covfitswp = '{}catalog/BOSS_zbins_wp/{}_log_z{}z{}_mocks_wp.fits.gz'.format(home,gal,zmin,zmax)
        colwp   = 'col1'
    pythonsel = (wp[:,0]>smin)&(wp[:,0]<smax)
    wp = wp[tuple(pythonsel),:]
    swp = wp[:,0]
    # observation
    obscfwp = Table.read(obs2pcfwp,format='ascii.no_header')
    selwp = (obscfwp[colwp]<smax)&(obscfwp[colwp]>=smin)
    OBSwp   = obscfwp['col4'][selwp]
    # Read the covariance matrices
    hdu = fits.open(covfitswp) 
    mockswp = hdu[1].data[GC+'mocks']#[binminwp:binmaxwp,:]
    Nmockwp = mockswp.shape[1] 
    errbarwp = np.std(mockswp,axis=1)
    hdu.close()  

    # include wp integraing to pimax with errorbars
    wppi = np.loadtxt('{}best-fit-wp_{}_{}-python_pi{}.dat'.format(fileroot[:-10],gal,GC,pimax))
    if (gal == 'LRG')|(gal=='ELG'):
        if rscale == 'linear':
            covfitswppi  = '{}catalog/nersc_{}_{}_{}/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,'wp',gal,ver,gal,zmin,zmax,pimax) 
            obs2pcfwppi  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}_pi{}.dat'.format(home,gal,ver,gal,GC,ver,pimax)
        elif rscale == 'log':
            covfitswppi = '{}catalog/nersc_zbins_wp_mps_{}/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,gal,gal,zmin,zmax,pimax) 
            obs2pcfwppi  = '{}catalog/nersc_zbins_wp_mps_{}/wp_rp_pip_eBOSS_{}_{}_{}_{}-{}_pi{}.dat'.format(home,gal,gal,GC,ver1,zmin,zmax,pimax)
        colwppi   = 'col3'
    else:
        obs2pcfwppi = '{}catalog/BOSS_zbins_wp/OBS_{}_NGC+SGC_DR12v5_z{}z{}_pi{}.wp'.format(home,gal,zmin,zmax,pimax)
        covfitswppi = '{}catalog/BOSS_zbins_wp/{}_log_z{}z{}_mocks_wp_pi{}.fits.gz'.format(home,gal,zmin,zmax,pimax)
        colwppi   = 'col1'
    pythonselpi = (wppi[:,0]>smin)&(wppi[:,0]<smax)
    wppi = wppi[tuple(pythonselpi),:]
    # observation
    obscfwppi = Table.read(obs2pcfwppi,format='ascii.no_header')
    selwppi = (obscfwppi[colwppi]<smax)&(obscfwppi[colwppi]>=smin)
    OBSwppi   = obscfwppi['col4'][selwppi]
    # Read the covariance matrices
    hdu = fits.open(covfitswppi) 
    mockswppi = hdu[1].data[GC+'mocks']#[binminwp:binmaxwp,:]
    Nmockwppi = mockswppi.shape[1] 
    errbarwppi = np.std(mockswppi,axis=1)
    hdu.close()  

    # SHAM halo catalogue
    if rscale=='linear':
        xi = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[binmin:binmax]
    else:
        xi = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[1:]
    errbarsham = np.loadtxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),usecols=(2,3))
    
    bbins,UNITv,SHAMv = np.loadtxt('{}/best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),unpack=True)
    pdf = SHAMv[:-1]/UNITv[:-1]
    
    #print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[2]))

    # data summary
    samples[zbin] = sample
    Ccodes[zbin] = xi
    obscfs[zbin] = obscf
    errbars[zbin] = errbar
    errbarshams[zbin] = errbarsham/np.sqrt(nseed)
    OBSwps[zbin] = OBSwp
    wps[zbin]     = wp    
    errbarwps[zbin] = errbarwp
    OBSwppis[zbin] = OBSwppi
    wppis[zbin]     = wppi    
    errbarwppis[zbin] = errbarwppi
    pdfs[zbin]  = pdf
    bestfits[zbin] = a.get_best_fit()
fontsize=12
if task == '2pcf':
    plt.rc('font', family='serif', size=fontsize) 
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(8,8))
    spec = gridspec.GridSpec(nrows=4,ncols=2, left = 0.12,right = 0.98,bottom=0.08,top = 0.96,height_ratios=[3, 1,3,1], hspace=0.,wspace=0.05)
    ax = np.empty((4,2), dtype=type(plt.axes))
    for zbin in range(zbinnum):
        if zbin !=zbinnum-1:
            K=0
        else: 
            K=1
            colors[zbin] = 'k'
            nbins = len(bins)-1
            s = (bins[:-1]+bins[1:])/2

        for col,covbin,name,k in zip(cols,[int(0),int(Nstot)],['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),obscfs[zbin][col]]
            err   = [np.ones(nbins),s**2*errbars[zbin][k*nbins:(k+1)*nbins]]

            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])
                ax[J,K].plot(s,s**2*(Ccodes[zbin][:,k+2]-values[j])/err[j],c=colors[zbin],alpha=0.6,label='${}<z<{}$'.format(zmins[zbin],zmaxs[zbin]))#,label='_hidden')
                ax[J,K].fill_between(s,s**2*(Ccodes[zbin][:,k+2]-values[j])/err[j]-s**2*errbarshams[zbin][:,k]/err[j],s**2*(Ccodes[zbin][:,k+2]-values[j])/err[j]+s**2*errbarshams[zbin][:,k]/err[j],color=colors[zbin],alpha=0.3,label='_hidden')

                if k==0:
                    ax[J,K].errorbar(s,s**2*(obscfs[zbin][col]-values[j])/err[j],s**2*errbars[zbin][k*nbins:(k+1)*nbins]/err[j],\
                        color=colors[zbin], marker='o',ecolor=colors[zbin],ls="none",\
                        label='_hidden',markersize=4)#'${}<z<{}$'.format(zmins[zbin],zmaxs[zbin]))
                else:
                    ax[J,K].errorbar(s,s**2*(obscfs[zbin][col]-values[j])/err[j],s**2*errbars[zbin][k*nbins:(k+1)*nbins]/err[j],\
                        color=colors[zbin], marker='o',ecolor=colors[zbin],ls="none",\
                        label='_hidden',markersize=4)#'${}<z<{}$'.format(zmins[zbin],zmaxs[zbin]))#, $\chi^2/dof={:.4}$/{},-2*bestfits[zbin]['log_likelihood'],int(2*len(s)-npar)))

                if rscale=='log':
                    plt.xscale('log')

                if (j==0):
                    if K==0:
                        ax[J,K].set_ylabel('$s^2 * \\xi_{}$'.format(k*2),fontsize=fontsize)
                    else: 
                        plt.yticks(alpha=0)

                    if k == 0:
                        if gal == 'LOWZ':
                            plt.ylim(91,130)
                        elif gal == 'CMASS':
                            plt.ylim(81,113)
                        elif gal == 'LRG':
                            plt.ylim(55,114)
                        plt.legend(loc=4,prop={"size":fontsize+2})
                    else:
                        if gal == 'LOWZ':
                            plt.ylim(-84,37)
                        elif gal == 'CMASS':
                            plt.ylim(-79,28)
                        elif gal == 'LRG':
                            plt.ylim(-84,45)
                else:
                    if K==0:
                        ax[J,K].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2),fontsize=fontsize)
                    else: 
                        plt.yticks([])
                    plt.ylim(-3,3)
                if J==3:
                    plt.xticks([5,10,15,20,25])
                    plt.xlabel('s ($h^{-1}$Mpc)',fontsize=fontsize)
                else:
                    plt.xticks([])

    plt.savefig('{}cf_{}_{}bestfit_{}_{}_{}-{}Mpch-1_new.png'.format(home,multipole,date,gal,GC,rmin,rmax))
    plt.close()
elif task == 'wp':
    # plot the wp vs wp with pimax=25
    plt.rc('font', family='serif', size=fontsize) 
    fig = plt.figure(figsize=(7,7))
    spec = gridspec.GridSpec(nrows=4,ncols=2, left = 0.12,right = 0.98,bottom=0.1,top = 0.98,height_ratios=[2, 1,2,1], hspace=0,wspace=0.05)
    ax = np.empty((4,2), dtype=type(plt.axes))
    for zbin in range(zbinnum):
        if zbin !=zbinnum-1:
            K=0
        else: 
            K=1
            colors[zbin] = 'k'

        for k,name in enumerate(['pi80','pi30']):
            obswptype = [OBSwps[zbin],OBSwppis[zbin]]
            #import pdb;pdb.set_trace()
            obswperrtype = [errbarwps[zbin],errbarwppis[zbin]]
            shamwp = [wps[zbin],wppis[zbin]]
            values=[np.zeros_like(OBSwp),obswptype[k]]
            err   = [np.ones_like(OBSwp),obswperrtype[k]]

            for j in range(2):
                J = 2*k+j
                ax[J,K] = fig.add_subplot(spec[J,K])              
                ax[J,K].plot(swp,(shamwp[k][:,1]-values[j])/err[j],c=colors[zbin],alpha=0.6,label='${}<z<{}$'.format(zmins[zbin],zmaxs[zbin]))
                ax[J,K].fill_between(swp,(shamwp[k][:,1]-values[j])/err[j]-shamwp[k][:,2]/err[j]/np.sqrt(nseed),\
                                    (shamwp[k][:,1]-values[j])/err[j]+shamwp[k][:,2]/err[j]/np.sqrt(nseed),\
                                    color=colors[zbin],alpha=0.3,label='_hidden')

                ax[J,K].errorbar(swp,(values[1]-values[j])/err[j],err[1]/err[j],\
                        color=colors[zbin], marker='o',ecolor=colors[zbin],ls="none",\
                        label='_hidden',markersize=3)
                plt.xscale('log')#,subsx=[5,10,25])
                

                if (j==0):
                    if zbin >=zbinnum-2:
                        ax[J,K].text(5,5,'$\{}_{{max}} = {} h^{{-1}}$Mpc'.format(name[:2],name[2:]))
                    plt.yscale('log')

                    if K==0:
                        ax[J,K].set_ylabel('$w_p$',fontsize=fontsize)
                        plt.xticks([])
                    else:
                        plt.yticks(alpha=0)
                    """
                    if gal == 'LOWZ':
                        plt.ylim(2,59)
                    elif gal == 'CMASS':
                        plt.ylim(2,57)
                    elif gal == 'LRG':
                        plt.ylim(2,57)
                    """
                    plt.ylim(4,100)
                    if gal == 'LRG':
                        if k==0:
                            plt.legend(loc=1,prop={"size":fontsize-2})#,\
                    else:
                        if k==0:
                            plt.legend(loc=1,prop={"size":fontsize+3})                        
                        #loc='upper center', bbox_to_anchor=(0.5, 5),ncol=2, fancybox=True, shadow=True)
                else:
                    if K==0:
                        ax[J,K].set_ylabel('$\Delta$ $w_p$/err',fontsize=fontsize)
                        plt.xticks([])
                    else:
                        plt.yticks(alpha=0)
                    if gal == 'LRG':
                        plt.ylim(-4,4)
                        plt.yticks([-3,0,3])
                    else:
                        plt.ylim(-2.5,2.5)
                        plt.yticks([-2,0,2])

                if J==3:
                    plt.xlabel('$r_p$ ($h^{{-1}}$Mpc)',fontsize=fontsize)

                from matplotlib.ticker import ScalarFormatter, NullFormatter
                for AX,axis in enumerate([ax[J,K].xaxis,ax[J,K].yaxis]):
                    axis.set_major_formatter(ScalarFormatter())
                    axis.set_minor_formatter(NullFormatter())
                    if (AX == 0)&(j==1):
                        ax[J,K].set_xticks([5,10,25])
                    elif (AX==1)&(j==0):
                        ax[J,K].set_yticks([5,10,20,40,80])
                
    plt.savefig('{}wp_{}_{}bestfit_{}_{}_{}-{}Mpch-1.png'.format(home,multipole,date,gal,GC,rmin,rmax))
    plt.close()
elif task == 'posteriors':        
    plt.rcParams['text.usetex'] = False
    g = plots.getSinglePlotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.2

    g = plots.getSubplotPlotter(subplot_size=3)
    g.settings.axes_fontsize=15
    g.settings.legend_fontsize = 18
    g.settings.lab_fontsize = 18
    g.triangle_plot(samples[:-1],parameters,filled=True,\
        legend_labels=['${}<z<{}$'.format(zmins[x],zmaxs[x]) for x in range(zbinnum-1)],\
            contour_colors=colors[:zbinnum])
    for zbin in range(zbinnum-1):
        for yi in range(npar): 
            for xi in range(yi):
                ax = g.subplots[yi,xi]
                ax.axvline(bestfits[zbin]['parameters'][xi], color=colors[zbin],ls = '--',alpha=0.5)
                ax.axhline(bestfits[zbin]['parameters'][yi], color=colors[zbin],ls ='--',alpha=0.5)
                ax.plot(bestfits[zbin]['parameters'][xi],bestfits[zbin]['parameters'][yi], "*",markersize=5,color='k') 
    g.export('{}ztot_{}_{}_{}_posterior.png'.format(home,date,gal,GC))
    plt.close()