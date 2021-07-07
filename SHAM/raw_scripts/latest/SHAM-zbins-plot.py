#!/usr/bin/env python3
import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.theory.wp import wp
import os
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import pymultinest
import corner
import h5py

# variables
gal      = sys.argv[1]
GC       = sys.argv[2]
rscale   = sys.argv[3] #'linear' # 'log'
function = 'mps' # 'wp'
zmin     = sys.argv[4]
zmax     = sys.argv[5]
finish   = int(sys.argv[6])
nseed    = 15
date     = '0218'
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
if rscale =='linear':
    rmax = 25
else:
    rmax = 30
nthread  = 1
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=30
home     = '/home/astro/jiayu/Desktop/SHAM/'
fileroot = '{}MCMCout/zbins_{}/{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,function,rscale,gal,GC,zmin,zmax)
cols = ['col4','col5']

# read the posterior file
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

# getdist results
sample = loadMCSamples(fileroot)
print('chain Results:')
stats = sample.getMargeStats()
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
sigma = np.zeros(npar)
for i in range(npar):
    par = stats.parWithName(parameters[i])
    mean[i] = par.mean
    sigma[i] = par.err
    lower[i] = par.limits[0].lower
    upper[i] = par.limits[0].upper
    best[i] = (lower[i] + upper[i]) * 0.5
    print('getdist {0:s}: [{1:.6f}, {2:.6f}]'.format( \
        parameters[i],  lower[i], upper[i]))
#getdist plot
plt.rcParams['text.usetex'] = False
g = plots.getSinglePlotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
g = plots.getSubplotPlotter()
g.triangle_plot(sample,parameters, filled=True)
for yi in range(npar): 
    for xi in range(yi):
        ax = g.subplots[yi,xi]
        ax.plot(a.get_best_fit()['parameters'][xi],a.get_best_fit()['parameters'][yi], "*",color='k') 
g.export('{}{}_{}_{}_posterior.png'.format(fileroot[:-10],date,gal,GC))
plt.close()

# corner results
A=a.get_equal_weighted_posterior()
figure = corner.corner(A[:,:npar],labels=[r"$sigma$",r"$Vsmear$", r"$Vceil$"],\
                       show_titles=True,title_fmt=None)
axes = np.array(figure.axes).reshape((npar,npar))
for yi in range(npar): 
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(a.get_best_fit()['parameters'][xi], color="g")
        ax.axhline(a.get_best_fit()['parameters'][yi], color="g")
        ax.plot(a.get_best_fit()['parameters'][xi],a.get_best_fit()['parameters'][yi], "sg") 
plt.savefig('{}{}_posterior_check_{}_{}.png'.format(fileroot[:-10],date,gal,GC))
plt.close()
print('the best-fit parameters: sigma {},Vsmear {} km/s, Vceil {} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2]))
print('its chi2: {:.6}'.format(-2*a.get_best_fit()['log_likelihood']))

if finish: 
    # write the multinest/gedist analysis report
    file = '{}Vzsmear_report_{}_{}.txt'.format(fileroot[:-10],gal,GC)
    f = open(file,'a')
    f.write('{} {} multinest: \n'.format(gal,GC))
    f.write('(-2)* max loglike: {} \n'.format(-2*a.get_best_fit()['log_likelihood']))
    f.write('max-loglike params: {}\n'.format(a.get_best_fit()['parameters']))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('getdist 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
    f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s,Vceil {:.6}+{:.6}{:.6}km/s  \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2],upper[2]-a.get_best_fit()['parameters'][2],lower[2]-a.get_best_fit()['parameters'][2]))
    
    stats = a.get_stats()    
    for j in range(npar):
        lower[j], upper[j] = stats['marginals'][j]['1sigma']
        print('multinest {0:s}: [{1:.6f} {2:.6f}]'.format(parameters[j],  upper[j], lower[j]))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('multinest analyser results: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
    f.write('another way around: sigma {0:.6}+{1:.6}{2:.6}, sigma_smear {3:.6}+{4:.6}{5:.6}km/s,Vceil {6:.6}+{7:.6}{8:.6}km/s  \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2],upper[2]-a.get_best_fit()['parameters'][2],lower[2]-a.get_best_fit()['parameters'][2]))
    f.close()

    # start the final 2pcf, wp, Vpeak histogram, PDF
    if (rscale=='linear')&(function=='mps'):
        if gal == 'LRG':
            SHAMnum   = int(6.26e4)
            z = 0.7781
            a_t = '0.56220'
            ver = 'v7_2'
        elif gal=='ELG':
            SHAMnum   = int(2.93e5)
            z = 0.87364
            a_t = '0.53780'
            ver = 'v7'
            cols = ['col3','col4']
        elif gal=='CMASSLOWZTOT':
            SHAMnum = 208000
            z = 0.5609
            a_t = '0.64210'
        elif gal=='CMASS':
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
            obstool = 'PIP'
        else:
            obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.mps'.format(home,gal,zmin,zmax)
            covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
            obstool = ''
        
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
        
    elif (rscale=='log'):
        # read s bins
        binfile = Table.read(home+'binfile_log.dat',format='ascii.no_header');
        ver1='v7_2'
        sel = (binfile['col3']<rmax)&(binfile['col3']>=rmin)
        bins  = np.unique(np.append(binfile['col1'][sel],binfile['col2'][sel]))
        s = binfile['col3'][sel]
        nbins = len(bins)-1
        binmin = np.where(binfile['col3']>=rmin)[0][0]
        binmax = np.where(binfile['col3']<rmax)[0][-1]+1

        if gal == 'LRG':
            ver = 'v7_2'
        else:
            ver = 'v7'
        # filenames
        covfits = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,function,rscale,gal,zmin,zmax,multipole) 
        obs2pcf  = '{}catalog/nersc_zbins_wp_mps_{}/{}_{}_{}_{}_eBOSS_{}_zs_{}-{}.dat'.format(home,gal,function,rscale,gal,GC,ver,zmin,zmax)
        # Read the covariance matrices 
        hdu = fits.open(covfits) # cov([mono,quadru])
        mocks = hdu[1].data[GC+'mocks']
        Nmock = mocks.shape[1]
        hdu.close()
        # observations
        obscf = Table.read(obs2pcf,format='ascii.no_header')
        obscf= obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
        # prepare OBS, covariance and errobar for chi2
        Nstot = int(mocks.shape[0]/2)
        mocks = vstack((mocks[binmin:binmax,:],mocks[binmin+Nstot:binmax+Nstot,:]))
        covcut  = cov(mocks).astype('float32')
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')# LRG columns are s**2*xi
        covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
        print('the covariance matrix and the observation 2pcf vector are ready.')

        # zbins, z_eff ans ngal
        if (zmin=='0.6')&(zmax=='0.8'):
            if gal=='ELG':
                SHAMnum = int(3.26e5)
                z = 0.7136
            else:
                SHAMnum = int(8.86e4)
                z = 0.7051
            a_t = '0.58760'
        elif (zmin=='0.6')&(zmax=='0.7'):            
            SHAMnum = int(9.39e4)
            z = 0.6518
            a_t = '0.60080'
        elif zmin=='0.65':
            SHAMnum = int(8.80e4)
            z = 0.7273
            a_t = '0.57470'
        elif zmin=='0.9':
            SHAMnum = int(1.54e5)
            z = 0.9938
            a_t = '0.50320'
        elif zmin=='0.7':
            if gal=='ELG':
                SHAMnum = int(4.38e5)
                z = 0.8045# To be calculated
            else:
                SHAMnum = int(6.47e4)
                z=0.7968
            a_t = '0.54980'
        else:
            if gal=='ELG':
                SHAMnum = int(3.34e5)
                z = 0.9045 # To be calculated
            else:
                SHAMnum = int(3.01e4)
                z= 0.8777
            a_t = '0.52600'
        obstool = ''

    else:
        print('wrong 2pcf function input')


    # wp: log binning
    binfilewp = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
    selwp = (binfilewp['col3']<smax)&(binfilewp['col3']>=smin)
    binswp  = np.unique(np.append(binfilewp['col1'][selwp],binfilewp['col2'][selwp]))
    swp = binfilewp['col3'][selwp]
    nbinswp = len(binswp)-1

    # analytical RR
    mu = (np.linspace(0,mu_max,nmu+1)[:-1]+np.linspace(0,mu_max,nmu+1)[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))
    # Analytical RR calculation
    RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
    rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
    print('the analytical random pair counts are ready.')

    # cosmological parameters
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)

    # SHAM halo catalogue
    if os.path.exists('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC)):
        xi = np.loadtxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
        wp = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
        bbins,UNITv,SHAMv = np.loadtxt('{}best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),unpack=True)
    else:
        print('reading the UNIT simulation snapshot with a(t)={}'.format(a_t))  
        halofile = home+'catalog/UNIT_hlist_'+a_t+'.hdf5'        
        read = time.time()
        f=h5py.File(halofile,"r")
        if len(f["halo"]['Vpeak'][:])%2 ==1:
            datac = np.zeros((len(f["halo"]['Vpeak'][:])-1,5))
            for i,key in enumerate(f["halo"].keys()):
                datac[:,i] = (f["halo"][key][:])[:-1]
        else:
            datac = np.zeros((len(f["halo"]['Vpeak'][:]),5))
            for i,key in enumerate(f["halo"].keys()):
                datac[:,i] = f["halo"][key][:]
        f.close()        
        half = int32(len(datac)/2)
        print(len(datac))
        print('read the halo catalogue costs {:.6}s'.format(time.time()-read))

        # generate uniform random numbers
        print('generating uniform random number arrays...')
        uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
        uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 

        # SHAM application
        def sham_tpcf(uni,uni1,sigM,sigV,Mtrun):
            x00,x20,v0,n0,wp0= sham_cal(uni,sigM,sigV,Mtrun)
            x01,x21,v1,n1,wp1= sham_cal(uni1,sigM,sigV,Mtrun)
            return [append(x00,x01),append(x20,x21),(v0+v1)/2, (n0+n1)/2, append(wp0,wp1)]

        def sham_cal(uniform,sigma_high,sigma,v_high):
            # scatter Vpeak
            scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
            scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
            datav = datac[:,1]*scatter
            # select halos
            percentcut = int(len(datac)*v_high/100)
            LRGscat = datac[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
            datav = datav[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
            LRGscat = LRGscat[argpartition(-datav,percentcut)[percentcut:]]
            datav = datav[argpartition(-datav,percentcut)[percentcut:]]
            # binnning Vpeak of the selected halos
            n,BINS = np.histogram(LRGscat[:,1],range =(0,1500),bins=100)
            
            # transfer to the redshift space
            scathalf = int(len(LRGscat)/2)
            z_redshift  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
            z_redshift %=boxsize
            
            # Corrfunc 2pcf and wp
            DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],z_redshift,periodic=True, verbose=False,boxsize=boxsize)
            wp_dat = wp(boxsize,80,nthread,binswp,LRGscat[:,2],LRGscat[:,3],z_redshift)
            # calculate the 2pcf and the multipoles
            mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
            quad = mono * 2.5 * (3 * mu**2 - 1)
            # use sum to integrate over mu
            return [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,min(datav),n,wp_dat['wp']]

        # calculate the SHAM 2PCF
        with Pool(processes = nseed) as p:
            xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2]))))) 
        # xi0
        tmp = [xi1_ELG[a][0] for a in range(nseed)]
        true_array = np.hstack((((np.array(tmp)).T)[:nbins],((np.array(tmp)).T)[nbins:]))
        mean0 = np.mean(true_array,axis=1)
        std0  = np.std(true_array,axis=1)
        # xi2
        tmp = [xi1_ELG[a][1] for a in range(nseed)]
        true_array = np.hstack((((np.array(tmp)).T)[:nbins],((np.array(tmp)).T)[nbins:]))
        mean1 = np.mean(true_array,axis=1)
        std1  = np.std(true_array,axis=1)
        # merge 2pcf multipoles
        model = append(mean0,mean1)
        errsham = append(std0,std1)
        res = OBS-model
        print('python chi2 = {:.3f}'.format(res.dot(covR.dot(res))))

        # save python 2pcf
        xi = np.hstack((model.reshape(2,nbins).T,errsham.reshape(2,nbins).T))
        np.savetxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),xi,header='python chi2 = {:.3f}\n xi0 x12 xi0err xi2err'.format(res.dot(covR.dot(res))))

        # wp
        tmp = [xi1_ELG[a][4] for a in range(nseed)]
        true_array = np.hstack((((np.array(tmp)).T)[:nbinswp],((np.array(tmp)).T)[nbinswp:]))
        wp= (np.array([swp,np.mean(true_array,axis=1),np.std(true_array,axis=1)]).reshape(3,nbinswp)).T
        np.savetxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),wp,header='s wp wperr')

        # distributions:
        UNITv,b = np.histogram(datac[:,1],range =(0,1500),bins=100)
        SHAMv = np.mean(xi1_ELG,axis=0)[3]
        bbins = (b[1:]+b[:-1])/2
        bbins = np.append(bbins,np.inf)
        UNITv = np.append(UNITv,len(datac))
        SHAMv = np.append(SHAMv,SHAMnum)
        np.savetxt('{}best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),np.array([bbins,UNITv,SHAMv]).T,header='bins UNIT SHAM')

        # result report
        file = '{}Vzsmear_report_{}_{}.txt'.format(fileroot[:-10],gal,GC)
        f = open(file,'a')#;import pdb;pdb.set_trace()
        f.write('python chi2 = {:.3f}, correspond to Vceil = {:.6}km/s \n'.format(res.dot(covR.dot(res)),np.mean(xi1_ELG,axis=0)[2]))
        pdf = SHAMv[:-1]/UNITv[:-1]
        f.write('z{}z{} PDF max: {} km/s \n'.format(zmin,zmax,(bbins[:-1])[pdf==max(pdf[~np.isnan(pdf)])]))        
        f.close()

    # plot the 2pcf results
    errbar = np.std(mocks,axis=1)
    #print('mean Vceil:{:.3f}'.format(np.mean(xi1_ELG,axis=0)[2]))
    if rscale=='linear':
        Ccode = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[binmin:binmax]
    else:
        Ccode = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[1:]

    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(cols,[int(0),int(Nstot)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscf[col]]        
        err   = [np.ones(nbins),s**2*errbar[k*nbins:(k+1)*nbins]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            #ax[j,k].plot(s,s**2*(xi[:,k]-values[j]),c='c',alpha=0.6,label='SHAM-python')
            ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='m',alpha=0.6,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-3)))
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j])/err[j],s**2*errbar[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='{} obs 1$\sigma$'.format(obstool))
            plt.xlabel('s (Mpc $h^{-1}$)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                if k==0:
                    plt.legend(loc=2)
                else:
                    plt.legend(loc=1)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2))
                plt.ylim(-3,3)

    plt.savefig('{}cf_{}_bestfit_{}_{}_z{}z{}_{}-{}Mpch-1.png'.format(fileroot[:-10],multipole,gal,GC,zmin,zmax,rmin,rmax),bbox_tight=True)
    plt.close()

    # plot the 2PCF multipoles 2-25Mpc/h
    if rscale == 'linear':
        Ccode = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[2:binmax]
        # Read the covariance matrices and observations
        hdu = fits.open(covfits) #
        mock = hdu[1].data[GC+'mocks']
        Nmock = mock.shape[1] 
        hdu.close()
        binmin = 2
        nbins = binmax-binmin
        bins  = np.arange(binmin,binmax+1,1)
        s = (bins[:-1]+bins[1:])/2
        mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
        errbar = np.std(mocks,axis=1)
        covcut  = cov(mocks).astype('float32')
        obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
        if gal == 'ELG':
            OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
        else:
            OBS   = append(obscf['col4'],obscf['col5']).astype('float32')
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(cols,[int(0),int(Nstot)],['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),obscf[col]]        
            err   = [np.ones(nbins),s**2*errbar[k*nbins:(k+1)*nbins]]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                #ax[j,k].plot(s,s**2*(xi[:,k]-values[j]),c='c',alpha=0.6,label='SHAM-python')
                ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='m',alpha=0.6,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*(len(s)-3)-3)))
                ax[j,k].errorbar(s,s**2*(obscf[col]-values[j])/err[j],s**2*errbar[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='{} obs 1$\sigma$'.format(obstool))
                plt.xlabel('s (Mpc $h^{-1}$)')
                if rscale=='log':
                    plt.xscale('log')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                    if k==0:
                        plt.legend(loc=2)
                    else:
                        plt.legend(loc=1)
                    plt.title('correlation function {} at {}<z<{}: {} in {}'.format(name,zmin,zmax,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2))
                    plt.ylim(-3,3)

        plt.savefig('{}cf_{}_bestfit_{}_{}_{}-{}Mpch-1.png'.format(fileroot[:-10],multipole,gal,GC,2,rmax),bbox_tight=True)
        plt.close()


    # plot the Vpeak histogram 
    fig,ax = plt.subplots()
    plt.plot(bbins[:-1],SHAMv[:-1]/SHAMv[-1],color='b',label='SHAM')
    plt.plot(bbins[:-1],UNITv[:-1]/UNITv[-1],color='k',label='UNIT')
    plt.legend(loc=1)
    plt.xlim(0,1500)
    plt.ylim(1e-5,1)
    plt.yscale('log')
    plt.ylabel('frequency')
    plt.xlabel('Vpeak (km/s)')
    plt.savefig(fileroot[:-10]+'best_SHAM_Vpeak_hist_{}_{}.png'.format(gal,GC))
    plt.close()
    # plot the PDF log scale
    fig,ax = plt.subplots()
    plt.plot(bbins[:-1],SHAMv[:-1]/UNITv[:-1])
    plt.xlim(0,1500)
    plt.ylim(1e-5,1.0)
    plt.yscale('log')
    plt.ylabel('prob of having a galaxy')
    plt.xlabel('Vpeak (km/s)')
    plt.savefig(fileroot[:-10]+'best_SHAM_PDF_hist_{}_{}_log.png'.format(gal,GC))
    plt.close()
    # plot the PDF linear scale
    fig,ax = plt.subplots()
    plt.plot(bbins[:-1],SHAMv[:-1]/UNITv[:-1])
    plt.xlim(0,1500)
    if gal =='LRG':
        plt.ylim(0,0.2)
    elif gal == 'ELG':
        plt.ylim(0,0.02)
    plt.ylabel('prob of having a galaxy')
    plt.xlabel('Vpeak (km/s)')
    plt.savefig(fileroot[:-10]+'best_SHAM_PDF_hist_{}_{}.png'.format(gal,GC))
    plt.close()
    pdf = SHAMv[:-1]/UNITv[:-1]
    print('z{}z{} PDF max: {} km/s'.format(zmin,zmax,(bbins[:-1])[pdf==max(pdf[~np.isnan(pdf)])]))


    # plot wp with errorbars
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

    # plot the wp
    errbarwp = np.std(mockswp,axis=1)
    fig = plt.figure(figsize=(6,7))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    for k in range(1):
        values=[np.zeros_like(OBSwp),OBSwp]
        err   = [np.ones_like(OBSwp),errbarwp]

        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])#;import pdb;pdb.set_trace()
            ax[j,k].errorbar(swp,(OBSwp-values[j])/err[j],errbarwp/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs 1$\sigma$ $\pi$80')
            ax[j,k].plot(swp,(wp[:,1]-values[j])/err[j],color='b',label='SHAM $\pi$80')
            plt.xlabel('rp (Mpc $h^{-1}$)')
            plt.xscale('log')
            if (j==0):        
                plt.yscale('log')
                ax[j,k].set_ylabel('wp')
                plt.legend(loc=0)
                plt.title('projected 2pcf at {}<z<{}: {} in {}'.format(zmin,zmax,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta$ wp/err')
                plt.ylim(-3,3)

    plt.savefig('{}wp_bestfit_{}_{}_{}-{}Mpch-1_pi80.png'.format(fileroot[:-10],gal,GC,smin,smax),bbox_tight=True)
    plt.close()
