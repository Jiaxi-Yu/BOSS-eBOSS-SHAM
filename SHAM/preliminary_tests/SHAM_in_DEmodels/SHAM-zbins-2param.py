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
from halotools.mock_observables import s_mu_tpcf
from halotools.mock_observables import tpcf_multipole
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
cataname = sys.argv[1]
mode = sys.argv[2]
gal      = 'CMASSLOWZTOT'
GC       = 'NGC+SGC'
rscale   = 'linear' # 'log'
function = 'mps' # 'wp'
zmin     = '0.2'
zmax     = '0.75'
nseed    = 30
date     = '0218'
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 400
rmin     = 5
rmax = 25
nthread  = 1
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
smin=5; smax=35
home     = '/home/astro/jiayu/Desktop/SHAM/'
cols = ['col4','col5']
finish = False

# start the final 2pcf, wp, Vpeak histogram, PDF
SHAMnum = int(208000/(1000/boxsize)**3)
scathalf = int(SHAMnum/2)
z = 0.5609
a_t = '0.64210'
# generate s bins
bins  = np.arange(rmin,rmax+1,1)
nbins = len(bins)-1
binmin = rmin
binmax = rmax
s = (bins[:-1]+bins[1:])/2
# covariance matrices and observations
obs2pcf = '{}catalog/BOSS_zbins_mps/OBS_{}_NGC+SGC_DR12v5_z{}z{}.mps'.format(home,gal,zmin,zmax)
covfits  = '{}catalog/BOSS_zbins_mps/{}_{}_z{}z{}_mocks_{}.fits.gz'.format(home,gal,rscale,zmin,zmax,multipole)
obstool = ''
    
# Read the covariance matrices and observations
hdu = fits.open(covfits) #

##############################################
mock = hdu[1].data[GC+'mocks']*10 # *10 considering the boxsize
##############################################
Nmock = mock.shape[1]
hdu.close()
Nstot=100
mocks = vstack((mock[binmin:binmax,:],mock[binmin+Nstot:binmax+Nstot,:]))
covcut  = cov(mocks).astype('float32') 
obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]       
OBS   = append(obscf['col4'],obscf['col5']).astype('float32')            
covR  = np.linalg.pinv(covcut)*(Nmock-len(mocks)-2)/(Nmock-1)
print('the covariance matrix and the observation 2pcf vector are ready.')

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
print('reading the UNIT simulation snapshot with a(t)={}'.format(a_t))  
halofile = home+'catalog/'+cataname+'_snapshot_009.z0.500.AHF_halos.hdf5'       
read = time.time()
f=h5py.File(halofile,"r")
if len(f["halo"]['Vmax'][:])%2 ==1:
    datac = np.zeros((len(f["halo"]['Vmax'][:])-1,5))
    for i,key in enumerate(f["halo"].keys()):
            datac[:,i] = (f["halo"][key][:])[:-1]
else:
    datac = np.zeros((len(f["halo"]['Vmax'][:]),5))
    for i,key in enumerate(f["halo"].keys()):
        datac[:,i] = f["halo"][key][:]
f.close()        
datac[:,2:]/=1000
half = int32(len(datac)/2)
print(len(datac))
print('read the halo catalogue costs {:.6}s'.format(time.time()-read))

# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac)).astype('float32') for x in range(nseed)] 


# SHAM application
def sham_tpcf(uni,uni1,sigM,sigV):
    if finish:
        return sham_cal(uni,sigM,sigV)
    else:
        x00,x20= sham_cal(uni,sigM,sigV)
        x01,x21= sham_cal(uni1,sigM,sigV)
        return [(x00+x01)/2,(x20+x21)/2]

def sham_cal(uniform,sigma_high,sigma):
    # scatter Vpeak
    scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
    scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
    datav = datac[:,1]*scatter
    # select halos
    LRGscat = datac[argpartition(-datav,SHAMnum)[:(SHAMnum)]]
    
    # transfer to the redshift space
    z_redshift  = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    z_redshift %=boxsize
    
    # Corrfunc 2pcf and wp
    LRGscat[:,-1] = z_redshift*1
    """
    np.savetxt('./tmp/{:.5}.dat'.format(uniform[0]),LRGscat[:,2:])
    os.system('../FCFC/FCFC_2PT_BOX --conf fcfc_2pt_box.conf -i ./tmp/{:.5}.dat -P ./tmp/{:.5}.dd -E ./tmp/{:.5}.xi -M ./tmp/{:.5}.mps'.format(uniform[0],uniform[0],uniform[0],uniform[0]))
    mono,quad = np.loadtxt('./tmp/{:.5}.mps'.format(uniform[0]),unpack=True)[3:]
    os.system('rm ./tmp/{:.5}.dat'.format(uniform[0]))
    os.system('rm ./tmp/{:.5}.dd'.format(uniform[0]))
    os.system('rm ./tmp/{:.5}.xi'.format(uniform[0]))
    """
    mu1 = np.linspace(0,mu_max,nmu+1)
    xi_s_mu = s_mu_tpcf(LRGscat[:,2:],bins, mu1, period=boxsize, num_threads=nthread)
    xi0 = tpcf_multipole(xi_s_mu, mu1, order=0)
    xi2 = tpcf_multipole(xi_s_mu, mu1, order=2)
    """    
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],LRGscat[:,4],periodic=True, verbose=False,boxsize=boxsize)
    mon = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
    qua = mon * 2.5 * (3 * mu**2 - 1)
    mono,quad  = np.sum(mon,axis=-1)/nmu, np.sum(qua,axis=-1)/nmu
    print('mono diff:',(mono-xi0)/xi0)
    print('quad diff:',(quad-xi2)/xi2)
    """

    # use sum to integrate over mu
    if finish:
        return LRGscat
    else:
        return [xi0,xi2]
# chi2
def chi2(sigma_M,sigma_V):
# calculate mean monopole in parallel
    with Pool(processes = nseed) as p:
        xi0_tmp = p.starmap(sham_tpcf,zip(uniform_randoms,uniform_randoms1,repeat(float32(sigma_M)),repeat(float32(sigma_V))))
    # average the result for multiple seeds
    xi0,xi2 = mean(xi0_tmp,axis=0,dtype='float32')[0],\
                mean(xi0_tmp,axis=0,dtype='float32')[1]
    model = append(xi0,xi2)
    # calculate the residuals and chi2
    res = OBS-model
    return res.dot(covR.dot(res))

# read the posterior file
parameters = ["sigma","Vsmear"]
npar = len(parameters)
fileroot = '{}MCMCout/zbins_{}/{}_Vsmear/multinest_'.format(home,'python',cataname)

# prior
def prior(cube, ndim, nparams):
    cube[0] = cube[0]
    cube[1] = 100*cube[1]+100

# loglikelihood = -0.5*chi2    
def loglike(cube, ndim, nparams):
    return -0.5*chi2(cube[0],cube[1])   
    
if mode == 'debug':
    print('debug mode on')
    T0 = time.time()
    chisq = chi2(0.5009077426917794, 145.38649650635975)
    print('chi2 = {:.3f}, time {:.3} s'.format(chisq,time.time()-T0))
else:
    # run MultiNest & write the parameter's name
    pymultinest.run(loglike, prior, npar,n_live_points= npoints, outputfiles_basename=fileroot, \
                    resume =True, verbose = False,n_iter_before_update=5,write_output=True)

    # analyse
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

    # cormer results
    A=a.get_equal_weighted_posterior()
    figure = corner.corner(A[:,:npar],labels=[r"$sigma$",r"$Vsmear$"],\
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
    print('the best-fit parameters: sigma {:.4},Vsmear {:.6} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1]))
    print('its chi2: {:.6}'.format(-2*a.get_best_fit()['log_likelihood']))

    # write the multinest/gedist analysis report
    stats = a.get_stats()    
    file = '{}Vzsmear_report_{}_{}.txt'.format(fileroot[:-10],gal,GC)
    f = open(file,'a')
    f.write('{} {} multinest: \n'.format(gal,GC))
    f.write('(-2)* max loglike: {} \n'.format(-2*a.get_best_fit()['log_likelihood']))
    f.write('max-loglike params: {}\n'.format(a.get_best_fit()['parameters']))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('getdist 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1]))
    f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1]))

    for j in range(npar):
        lower[j], upper[j] = stats['marginals'][j]['1sigma']
        print('getdist {0:s}: [{1:.6f} {2:.6f}]'.format(parameters[j],  upper[j], lower[j]))
    f.write('\n----------------------------------------------------------------------\n')
    f.write('multinest 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1]))
    f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1]))
    f.close()


    def sham_tpcf1(uni,uni1,sigM,sigV):
        x00,x20= sham_cal(uni,sigM,sigV)
        x01,x21= sham_cal(uni1,sigM,sigV)
        return [append(x00,x01),append(x20,x21)]

    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_tpcf1,list(zip(uniform_randoms,uniform_randoms1,\
        repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])))))
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

    # save one of the best-fit catalogue
    finish = True
    catalog = sham_cal(uniform_randoms[0],a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1])
    np.savetxt('{}best-fit_catalog-python.dat'.format(fileroot[:-10]),xi,header='Vz Vpeak X Y Z')

    # plot the results
    errbar = np.std(mocks,axis=1)
    stdSHAM = [std0,std1]
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], hspace=0.3,wspace=0.4)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,covbin,name,k in zip(cols,[int(0),int(Nstot)],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscf[col]]        
        err   = [np.ones(nbins),s**2*errbar[k*nbins:(k+1)*nbins]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j])/err[j],s**2*errbar[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='{} obs 1$\sigma$'.format(obstool))
            plt.xlabel('s (Mpc $h^{-1}$)')
            if rscale=='log':
                plt.xscale('log')
            if (j==0):
                ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2))#('\\xi_{}$'.format(k*2))#
                if k==0:
                    ax[j,k].errorbar(s+0.2,s**2*(xi[:,k]-values[j]),s**2*stdSHAM[k],c='c', marker='^',ecolor='c',ls="none",alpha=0.8,label='SHAM')
                    plt.legend(loc=2)
                else:
                    ax[j,k].errorbar(s+0.2,s**2*(xi[:,k]-values[j]),s**2*stdSHAM[k],c='c', marker='^',ecolor='c',ls="none",alpha=0.8,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*(len(s)-3)-3)))
                    plt.legend(loc=1)
                plt.title('correlation function {}: {} in {}'.format(name,gal,GC))
            if (j==1):
                ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2))
                ax[j,k].plot(s,s**2*(xi[:,k]-values[j])/err[j],c='c',alpha=0.8)
                plt.ylim(-3,3)

    plt.savefig('{}cf_{}_bestfit_{}_{}_{}-{}Mpch-1.png'.format(fileroot[:-10],multipole,gal,GC,rmin,rmax),bbox_tight=True)
    plt.close()
