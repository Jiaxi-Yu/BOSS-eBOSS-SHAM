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
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec
from getdist import plots, MCSamples, loadMCSamples
import sys
import corner
import h5py

# variables
gal      = sys.argv[1]
GC       = 'SGC'
func     = sys.argv[2]
cut      = sys.argv[3] #index, dsigma, posdsigma
nseed    = 15
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
home     = '/global/cscratch1/sd/jiaxi/SHAM/'
direc    = '/global/homes/j/jiaxi/'
mode    = 'best_check'
cols = ['col4','col5']

# covariance matrix and the observation 2pcf path
if gal == 'LRG':
    SHAMnum   = int(6.26e4)
    zmin     = 0.6
    zmax     = 1.0
    z = 0.7018
    ver='v7_2'
    halofile = home+'catalog/UNIT_hlist_0.58760.hdf5' 
    fileroot = direc+'MCMCout/indexcut_1027/LRG_SGC_nseed40/multinest_'
elif gal == 'ELG':
    SHAMnum   = int(2.93e5)
    zmin     = 0.6
    zmax     = 1.1
    z = 0.8594
    ver='v7'
    halofile = home+'catalog/UNIT_hlist_0.53780.hdf5'
    fileroot = direc+'MCMCout/indexcut_1211/mps_ELG_SGC_5-25/multinest_'
    cols = ['col3','col4']
    
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

# generate separation bins

if func  == 'wp':
    rmin     = 0.5
    rmax     = 5
# zbins with log binned mps and wp
    covfits  = '{}catalog/nersc_{}_{}_{}/{}_{}_mocks.fits.gz'.format(direc,func,gal,ver,func,gal) 
    obs2pcf  = '{}catalog/nersc_wp_{}_{}/wp_rp_pip_eBOSS_{}_{}_{}.dat'.format(direc,gal,ver,gal,GC,ver)
    # bin
    binfile = Table.read(home+'binfile_CUTE.dat',format='ascii.no_header')
    binmin = np.where(binfile['col3']>=rmin)[0][0]
    binmax = np.where(binfile['col3']<rmax)[0][-1]+1
    # observation
    obscf = Table.read(obs2pcf,format='ascii.no_header')
    obscf = obscf[(obscf['col3']<rmax)&(obscf['col3']>=rmin)]
    OBS   = obscf['col4']
    bins  = np.unique(append(obscf['col1'],obscf['col2']))
    nbins = len(bins)-1
    s = obscf['col3']
    OBS   = np.array(obscf['col4']).astype('float32')
    # Read the covariance matrices
    hdu = fits.open(covfits) 
    mocks = hdu[1].data[GC+'mocks'][binmin:binmax,:]
    Nmock = mocks.shape[1] 
    errbar = np.std(mocks,axis=1)
    hdu.close()
    covcut  = cov(mocks).astype('float32')
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
elif func == 'mps':
    rmin     = 5
    rmax     = 25
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    obs2pcf = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_{}.dat'.format(direc,gal,ver,func,rscale,gal,GC)
    covfits  = '{}catalog/nersc_mps_{}_{}/{}_{}_{}_mocks_{}.fits.gz'.format(direc,gal,ver,func,rscale,gal,multipole)
    # Read the covariance matrices and observations
    hdu = fits.open(covfits) #
    mock = hdu[1].data[GC+'mocks']
    Nmock = mock.shape[1] 
    hdu.close()
    mocks = vstack((mock[binmin:binmax,:],mock[binmin+200:binmax+200,:]))
    covcut  = cov(mocks).astype('float32')
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
    if gal == 'LRG':
        OBS   = append(obscf['col4'],obscf['col5']).astype('float32')
    else:
        OBS   = append(obscf['col3'],obscf['col4']).astype('float32')
        
    covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
print('the covariance matrix and the observation 2pcf vector are ready.')
    
# create the halo catalogue and plot their 2pcf
A = glob.glob('{}best-fit_{}_{}-wp_test_*_{}-{}Mpch-1.dat'.format(fileroot[:-10],gal,GC,rmin,rmax))
if len(A)==4:
    errbar = np.std(mocks,axis=1)
    if func =='mps':
        xi = [np.loadtxt('{}best-fit_{}_{}-python{}.dat'.format(fileroot[:-10],gal,GC,x)) for x in range(len(A))]
        V = [np.loadtxt('{}best-fit_Vpeak_hist_{}_{}{}.dat'.format(fileroot[:-10],gal,GC,y))[:,2] for y in range(len(A))]
        Vbin = np.loadtxt('{}best-fit_Vpeak_hist_{}_{}{}.dat'.format(fileroot[:-10],gal,GC,0))[:,0]
        UNIT = np.loadtxt('{}best-fit_Vpeak_hist_{}_{}{}.dat'.format(fileroot[:-10],gal,GC,0))[:,1]
        # 2pcf and Vpeak distribution plot
        fig = plt.figure(figsize=(14,8))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for col,covbin,name,k in zip(cols,[int(0),int(200)],['monopole','quadrupole'],range(2)):
            values=[np.zeros(nbins),obscf[col]]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                
                ax[j,k].plot(s,s**2*(xi[0][:,k]-values[j]),label='indexcut',c='b')
                ax[j,k].plot(s,s**2*(xi[1][:,k]-values[j]),label = 'Gaussian_scatter/$\sigma$ cut',c='orange')
                ax[j,k].plot(s,s**2*(xi[2][:,k]-values[j]),label = 'positive_scatter/$\sigma$ cut',c='green')                
                ax[j,k].errorbar(s,s**2*(obscf[col]-values[j]),s**2*errbar[k*nbins:(k+1)*nbins],color='k', marker='o',ecolor='k',ls="none",label='PIP obs')
                plt.xlabel('s (Mpc $h^{-1}$)')
                if (j==0):
                    ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))
                    plt.legend(loc=0)
                    plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
                if (j==1):
                    ax[j,k].set_ylabel('$s^2 * \Delta\\xi_{}$'.format(k*2))
        plt.savefig('cf_{}_bestfit-comp_{}_{}_{}-{}Mpch-1.png'.format(multipole,gal,GC,rmin,rmax))
        plt.close()

        # plot the histogram
        fig,ax = plt.subplots()
        plt.plot(Vbin[:-1],V[0][:-1]/V[0][-1],label='SHAM-indexcut')
        plt.plot(Vbin[:-1],V[1][:-1]/V[1][-1],label='SHAM-Gaussian_scatter/$\sigma$ cut')
        plt.plot(Vbin[:-1],V[2][:-1]/V[2][-1],label='SHAM-positive_scatter/$\sigma$ cut')
        plt.plot(Vbin[:-1],UNIT[:-1]/UNIT[-1],color='k',label='UNIT')
        plt.legend(loc=1)
        plt.xlim(0,1500)
        plt.ylim(1e-5,1)
        plt.yscale('log')
        plt.ylabel('frequency')
        plt.xlabel('Vpeak (km/s)')
        plt.legend(loc=0)
        plt.savefig('best_SHAM_Vpeak_hist-comp_{}_{}_comp.png'.format(gal,GC))
        plt.close()

        fig,ax = plt.subplots()
        plt.plot(Vbin[:-1],V[0][:-1]/UNIT[:-1],label='SHAM_indexcut')
        plt.plot(Vbin[:-1],V[1][:-1]/UNIT[:-1],label='SHAM-Gaussian_scatter/$\sigma$ cut')
        plt.plot(Vbin[:-1],V[2][:-1]/UNIT[:-1],label='SHAM-positive_scatter/$\sigma$ cut')
        plt.xlim(0,1500)
        plt.ylim(1e-5,1.0)
        plt.yscale('log')
        plt.legend(loc=0)
        plt.ylabel('prob of having a galaxy')
        plt.xlabel('Vpeak (km/s)')
        plt.savefig('best_SHAM_PDF_hist-comp_{}_{}_comp.png'.format(gal,GC))
        plt.close()

    elif func =='wp':
        wp = [np.loadtxt('{}best-fit_{}_{}-wp_test_{}_{}-{}Mpch-1.dat'.format(fileroot[:-10],gal,GC,x,rmin,rmax)) for x in ['index','dsigma','posdsigma','pos']]
        res = [obscf['col4']-np.loadtxt('{}best-fit_{}_{}-wp_test_{}_{}-{}Mpch-1.dat'.format(fileroot[:-10],gal,GC,x,rmin,rmax))[:,1] for x in ['index','dsigma','posdsigma','pos']]
        chi = [res1.dot(covR.dot(res1)) for res1 in res]
        print('index, dsigma, posdsigma reduced chi2 = {:.3f}, {:.3f}, {:.3f}/{}'.format(chi[0],chi[1],chi[2],nbins))
        #print(binmin,binmax,wp,obscf['col4'])

        # plot the 2PCF multipoles   
        fig = plt.figure(figsize=(5,6))
        spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.15)
        ax = np.empty((2,1), dtype=type(plt.axes))
        values=[np.zeros(nbins),obscf['col4']]
        err   = [np.ones(nbins),errbar]
        k=0
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s*1.02,(wp[0][:,1]-values[j])/err[j],wp[0][:,2]/err[j],alpha=0.7,label='indexcut',c='b')
            ax[j,k].errorbar(s*1.01,(wp[1][:,1]-values[j])/err[j],wp[1][:,2]/err[j],alpha= 0.7,label = 'Gaussian_scatter/$\sigma$ cut',c='orange')
            ax[j,k].errorbar(s*0.99,(wp[2][:,1]-values[j])/err[j],wp[2][:,2]/err[j],alpha=0.7,label = 'positive_scatter/$\sigma$ cut',c='green')
            ax[j,k].errorbar(s*0.98,(wp[3][:,1]-values[j])/err[j],wp[3][:,2]/err[j],alpha=0.7,label = 'positive_scatter V cut',c='m')

               
            ax[j,k].errorbar(s,(obscf['col4']-values[j])/err[j],errbar/err[j],color='k', marker='o',ecolor='k',ls="none",label="PIP obs",markersize=3)
            
            plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('wp')
                plt.legend(loc=0)
                plt.yscale('log')
                plt.title('projected correlation function: {} in {}'.format(gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta wp$/err')
                plt.xlabel('rp (Mpc $h^{-1}$)')
                plt.ylim(-3,3)

        plt.savefig('cf_{}_bestfit_wp-comp_{}_{}_{}-{}Mpch-1.png'.format(multipole,gal,GC,rmin,rmax))
        plt.close()

else:
    print('selecting only the necessary variables...')
    f=h5py.File(halofile,"r")
    sel = f["halo"]['Vpeak'][:]>0
    if len(f["halo"]['Vpeak'][:][sel])%2 ==1:
        datac = np.zeros((len(f["halo"]['Vpeak'][:][sel])-1,5))
        for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = (f["halo"][key][:][sel])[:-1]
    else:
        datac = np.zeros((len(f["halo"]['Vpeak'][:][sel]),5))
        for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = f["halo"][key][:][sel]
    f.close()        
    half = int(len(datac)/2)

    # generate uniform random numbers
    print('generating uniform random number arrays...')
    uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
    uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 


    # HAM application
    def sham_tpcf(uni,uni1):
        x00    = sham_cal(uni)
        x01    = sham_cal(uni1)
        return append(x00,x01)

    def sham_cal(uniform):
        if cut == 'index': # index-cut
            if gal =='LRG':
                sigma_high,sigma,v_high = 0.5801523501823749, 114.37337407958519, 4.83480817332233
            else:
                sigma_high,sigma,v_high = 1.3026967093684083, 0.45258011273993637, 6.473072873326147
            datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
            LRGscat = datac[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
            datav = datav[argpartition(-datav,SHAMnum+int(10**v_high))[:(SHAMnum+int(10**v_high))]]
            LRGscat = LRGscat[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
            datav = datav[argpartition(-datav,int(10**v_high))[int(10**v_high):]]
        elif cut == 'dsigma': # dsigma-cut
            if gal =='LRG':
                sigma_high,sigma,v_high = 3.721235759969714, 98.62135312958426, 1309.558192833906
            else:
                sigma_high,sigma,v_high = 2.870659692158652, 8.502187770572425, 339.50615755726847
            datav = datac[:,1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:])))/sigma_high #0.5s
            # modified Vpeak_scat
            org3  = datac[(datav<v_high)]  # 4.89s
            LRGscat = org3[np.argpartition(-datav[(datav<v_high)],SHAMnum)[:(SHAMnum)]]
        elif cut == 'posdsigma': # pos_scatter-dsigma-cut
            if gal =='LRG':
                sigma_high,sigma,v_high = 3.743151452102828, 98.77904596734587, 1285.2190538329396
            else:
                sigma_high,sigma,v_high = 3.122822903887847, 3.5966624937193172, 334.49358717280734
            scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
            scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
            datav = datac[:,1]*scatter/sigma_high #0.5s
            # modified Vpeak_scat
            org3  = datac[(datav<v_high)]  # 4.89s
            LRGscat = org3[np.argpartition(-datav[(datav<v_high)],SHAMnum)[:(SHAMnum)]]
        elif cut == 'pos': # pos_scatter-dsigma-cut
            if gal =='LRG':
                sigma_high,sigma,v_high = 1.0595005878116677, 106.37904926461752, 1378.5789925400386
            else:
                sigma_high,sigma,v_high = None, None, None
            scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
            scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
            datav = datac[:,1]*scatter #0.5s
            # modified Vpeak_scat
            org3  = datac[(datav<v_high)]  # 4.89s
            LRGscat = org3[np.argpartition(-datav[(datav<v_high)],SHAMnum)[:(SHAMnum)]]
        else:
            print('wrong input!')

        # transfer to the redshift space
        scathalf = int(len(LRGscat)/2)
        LRGscat[:,-1]  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
        LRGscat[:,-1] %=boxsize

        # calculate the 2pcf of the SHAM galaxies
        # count the galaxy pairs and normalise them
        wp_dat = wp(boxsize,80,nthread,bins,LRGscat[:,2],LRGscat[:,3],LRGscat[:,-1])#,periodic=True, verbose=True)
        return wp_dat['wp']

    # calculate the SHAM 2PCF
    with Pool(processes = nseed) as p:
        xi0_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1)))
    #
    tmp = [xi0_ELG[a] for a in range(nseed)]
    true_array = np.hstack((((np.array(tmp)).T)[:nbins],((np.array(tmp)).T)[nbins:]))
    model0 = np.mean(true_array,axis=1)
    std0  = np.std(true_array,axis=1)

    res = OBS-model0
    print('python reduced chi2 = {:.3f}/{}'.format(res.dot(covR.dot(res)),nbins))
    # save python 2pcf
    np.savetxt('{}best-fit_{}_{}-wp_test_{}_{}-{}Mpch-1.dat'.format(fileroot[:-10],gal,GC,cut,rmin,rmax),np.array([s,model0,std0]).T,header='python chi2 = {:.3f}/{}\n rp wp'.format(res.dot(covR.dot(res)),nbins))
