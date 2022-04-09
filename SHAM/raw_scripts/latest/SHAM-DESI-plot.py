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
from scipy.stats import cauchy

output  = '/global/homes/j/jiaxi/SHAM/'
home    = output+'catalog/DESISV3/'


# variables
gal      = sys.argv[1]
cata     = 'UNIT'#sys.argv[2] # UNIT ABACUS
date     = '0218'#sys.argv[3]#
Vsmeartype= sys.argv[2] # Gaussian Lorentzian
zmin     = sys.argv[3]
zmax     = sys.argv[4]
maxdv    = sys.argv[5]
targetdate  = 'data-202112'
GC       = 'NGC+SGC'#sys.argv[2]
rscale   = 'linear'#sys.argv[3] #'linear' # 'log'
finish   = 1#int(sys.argv[5])
pre      = ''#sys.argv[4]
#'0218': 3-param, '0726':mock-SHAM 3-param, '0729': 2-param
function = 'mps' # 'wp'
nseed    = 16
npoints  = 100 
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
if cata == 'UNIT': 
    boxsize = 1000
    Om       = 0.31
elif cata == 'ABACUS':
    boxsize  = 2000
    Om       = 0.315192
nthread  = 3
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
if Vsmeartype == 'Gaussian':
    fileroot = '{}MCMCout/zbins_{}/DESItest_{}/multinest_'.format(output,date,gal)
else:
    fileroot = '{}MCMCout/zbins_{}/DESItest_{}_Lorentzian_z{}z{}/multinest_'.format(output,date,gal,zmin,zmax)
cols = ['col2','col3']
if date == '0218':
    parameters = ["sigma","Vsmear","Vceil"]
elif date == '0729':
    parameters = ["sigma","Vsmear"]
rmin     = 5
rmax     = 25

# read the posterior file
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
print(a.get_best_fit()['parameters'])
# getdist results
sample = loadMCSamples(fileroot)
print('chain Results:')
stats = sample.getMargeStats()
best = np.zeros(npar)
lower = np.zeros(npar)
upper = np.zeros(npar)
mean = np.zeros(npar)
median = np.zeros(npar)
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
        ax = g.subplots[yi,xi] # to print best-fit, use [yi,yi], but sigma is too narrow
        ax.plot(a.get_best_fit()['parameters'][xi],a.get_best_fit()['parameters'][yi], "*",color='k') 
g.export('{}{}_{}_{}_posterior.png'.format(fileroot[:-10],date,gal,GC))
plt.close()

# corner results
A=a.get_equal_weighted_posterior()
figure = corner.corner(A[:,:npar],labels=parameters,\
                        #[r"$sigma$",r"$Vsmear$", r"$Vceil$"],\
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
if (date=='0729'):
    print('the best-fit parameters: sigma {},Vsmear {} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1]))
else:
    print('the best-fit parameters: sigma {},Vsmear {} km/s, Vceil {} km/s'.format(a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2]))
print('its chi2: {:.6}'.format(-2*a.get_best_fit()['log_likelihood']))


if finish:
    # write the multinest/gedist analysis report
    file = '{}Vzsmear_report_{}_{}.txt'.format(fileroot[:-10],gal,GC)
    if not os.path.exists(file):
        f = open(file,'w')
        f.write('{} {} multinest: \n'.format(gal,GC))
        f.write('(-2)* max loglike: {} \n'.format(-2*a.get_best_fit()['log_likelihood']))
        f.write('max-loglike params: {}\n'.format(a.get_best_fit()['parameters']))
        f.write('\n----------------------------------------------------------------------\n')
        if (date=='0729'):
            f.write('getdist 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1]))
            f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1]))    
        else:
            f.write('getdist 1-sigma errors: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
            f.write('another way around: sigma {:.6}+{:.6}{:.6}, sigma_smear {:.6}+{:.6}{:.6}km/s,Vceil {:.6}+{:.6}{:.6}km/s  \n'.format(a.get_best_fit()['parameters'][0],upper[0]-a.get_best_fit()['parameters'][0],lower[0]-a.get_best_fit()['parameters'][0],a.get_best_fit()['parameters'][1],upper[1]-a.get_best_fit()['parameters'][1],lower[1]-a.get_best_fit()['parameters'][1],a.get_best_fit()['parameters'][2],upper[2]-a.get_best_fit()['parameters'][2],lower[2]-a.get_best_fit()['parameters'][2]))
        stats = a.get_stats()    
        for j in range(npar):
            lower[j], upper[j] = stats['marginals'][j]['1sigma']
            median[j] = stats['marginals'][j]['median']
            print('multinest {0:s}: [{1:.6f} {2:.6f}]'.format(parameters[j],  upper[j], lower[j]))
        f.write('\n----------------------------------------------------------------------\n')
        if (date=='0729'):
            f.write('multinest analyser results: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s  \n'.format(lower[0],upper[0],lower[1],upper[1]))
            f.write('another way around: sigma {0:.6}+{1:.6}{2:.6}, sigma_smear {3:.6}+{4:.6}{5:.6}km/s \n'.format(median[0],upper[0]-median[0],lower[0]-median[0],median[1],upper[1]-median[1],lower[1]-median[1]))
        else:
            f.write('multinest analyser results: sigma [{:.6},{:.6}], sigma_smear [{:.6},{:.6}] km/s, Vceil [{:.6},{:.6}] km/s \n'.format(lower[0],upper[0],lower[1],upper[1],lower[2],upper[2]))
            f.write('another way around: sigma {0:.6}+ {1:.6}{2:.6}, sigma_smear {3:.6}+{4:.6}{5:.6}km/s,Vceil {6:.6}+{7:.6}{8:.6}km/s  \n'.format(median[0],upper[0]-median[0],lower[0]-median[0],median[1],upper[1]-median[1],lower[1]-median[1],median[2],upper[2]-median[2],lower[2]-median[2]))
        f.close()

    # start the final 2pcf, wp, Vpeak histogram, PDF
    if gal == 'LRG':
        SHAMnum   = 1001760
        zmin = 0.6 ; zmax  = 1.1
        a_t = '0.54980'
        #maxdv = 300
    elif gal == 'LRGlow':
        SHAMnum   = 1187982
        zmin = 0.4 ; zmax  = 0.6
        a_t = '0.67120'
        #maxdv = 300
    elif gal=='ELG':
        SHAMnum   = 13542938
        zmin = 0.8 ; zmax  = 1.5
        a_t = '0.48140'
        #maxdv = 100
    elif gal=='BGS_BRIGHT':
        SHAMnum   = 4258386
        zmin = 0.2 ; zmax  = 0.4
        a_t = '0.74980'
        #maxdv = 150
    elif gal == 'QSO':
        SHAMnum = 126606
        zmin = 0.8 ; zmax  = 2.1
        a_t = '0.41230'
        #maxdv = 400
    # delta v bins
    vbins = np.arange(-maxdv,maxdv+1,5)

    # generate s bins
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax
    s = (bins[:-1]+bins[1:])/2

    # covariance matrices and observations
    obs2pcf = '{}{}/{}_z{}z{}.dat'.format(home,targetdate,gal,zmin,zmax)
    obscf = Table.read(obs2pcf,format='ascii.no_header')[binmin:binmax]
    OBS = np.append(obscf['col2'],obscf['col3'])
    covR  = np.loadtxt('{}{}/{}-covR_z{}z{}.dat'.format(home,targetdate,gal,zmin,zmax))
    print('the covariance matrix and the observation 2pcf vector are ready.')

    # analytical RR
    mu = (np.linspace(0,mu_max,nmu+1)[:-1]+np.linspace(0,mu_max,nmu+1)[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))
    # Analytical RR calculation
    RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
    rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
    print('the analytical random pair counts are ready.')

    # plot wp with errorbars
    obs2pcfwp  = np.loadtxt('{}{}/{}_z{}z{}.{}'.format(home,targetdate,gal,zmin,zmax,'wp'))
    selwp = (obs2pcfwp[:,0]>rmin)&(obs2pcfwp[:,0]<rmax)
    obs2pcfwp = obs2pcfwp[tuple(selwp),:]
    swp = obs2pcfwp[:,0]
    nbinswp = len(swp)
    sbins = np.copy(bins)
    # wp observation & error
    OBSwp   = obs2pcfwp[:,1]
    errbarwp = obs2pcfwp[:,2]

    # cosmological parameters
    #import pdb;pdb.set_trace()
    z = 1/float(a_t)-1
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)

    # SHAM halo catalogue
    if os.path.exists('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC)):
        xi = np.loadtxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
        wp = np.loadtxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))
        bbins,UNITv,SHAMv = np.loadtxt('{}best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),unpack=True)
    else:
        print('reading the UNIT simulation snapshot with a(t)={}'.format(a_t))  
        halofile = home+'UNIT_hlist_{}.hdf5'.format(a_t)         
        read = time.time()
        f=h5py.File(halofile,"r")
        if len(f["halo"]['Vpeak'][:])%2 ==1:
            datac = np.zeros((len(f["halo"]['Vpeak'][:])-1,len(f["halo"].keys())))
            for i,key in enumerate(f["halo"].keys()):
                datac[:,i] = (f["halo"][key][:])[:-1]
        else:
            datac = np.zeros((len(f["halo"]['Vpeak'][:]),len(f["halo"].keys())))
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
        def sham_tpcf(*par):
            if date == '0729':
                x00,x20,wp0,v0,n0= sham_cal(par[0],par[2],par[3])
                x01,x21,wp1,v1,n1= sham_cal(par[1],par[2],par[3])
            else:
                x00,x20,wp0,v0,n0= sham_cal(par[0],par[2],par[3],par[4])
                x01,x21,wp1,v1,n1= sham_cal(par[1],par[2],par[3],par[4])
            return [append(x00,x01),append(x20,x21), append(wp0,wp1),(v0+v1)/2, (n0+n1)/2]

        # SHAM application
        def sham_cal(*PAR):
            # scatter Vpeak
            if date == '0729':
                uniform,sigma_high,sigma = PAR
                v_high = 0
            else:
                uniform,sigma_high,sigma,v_high = PAR
            
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
            if Vsmeartype == 'Gaussian':
                z_redshift = (LRGscat[:,4]+(LRGscat[:,0]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
            else:
                lorentz = np.tan((2*uniform-1)*np.pi)*sigma/2
                while len(lorentz[abs(lorentz)>maxdv])>0:
                    uni_tmp = np.random.RandomState(seed=time.time()).rand(len(lorentz[abs(lorentz)>maxdv]))
                    lorentz[abs(lorentz)>maxdv] = np.tan((2*uni_tmp-1)*np.pi)*sigma/2
                z_redshift  = (LRGscat[:,4]+(LRGscat[:,0]+lorentz)*(1+z)/H)
            z_redshift %=boxsize
            
            # Corrfunc 2pcf and wp
            DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],z_redshift,periodic=True, verbose=False,boxsize=boxsize)
            wp_dat = wp(boxsize,40,nthread,sbins,LRGscat[:,2],LRGscat[:,3],z_redshift)
            # calculate the 2pcf and the multipoles
            mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
            quad = mono * 2.5 * (3 * mu**2 - 1)
            # use sum to integrate over mu
            return [np.sum(mono,axis=-1)/nmu,np.sum(quad,axis=-1)/nmu,wp_dat['wp'],min(datav),n]

        # calculate the SHAM 2PCF
        if date == '0729':
            with Pool(processes = nseed) as p:
                xi1_ELG = p.starmap(sham_tpcf,list(zip(uniform_randoms,uniform_randoms1,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])))))
        else:
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
        errsham = append(std0,std1)/np.sqrt(nseed*2)
        res = OBS-model
        print('python chi2 = {:.3f}'.format(res.dot(covR.dot(res))))

        # save python 2pcf
        xi = np.hstack((model.reshape(2,nbins).T,errsham.reshape(2,nbins).T))
        np.savetxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),xi,header='python chi2 = {:.3f}\n xi0 x12 xi0err xi2err'.format(res.dot(covR.dot(res))))
        
        # wp
        tmp = [xi1_ELG[a][2] for a in range(nseed)]
        true_array = np.hstack((((np.array(tmp)).T)[:nbinswp],((np.array(tmp)).T)[nbinswp:]))
        wp= (np.array([swp,np.mean(true_array,axis=1),np.std(true_array,axis=1)/np.sqrt(nseed*2)]).reshape(3,nbinswp)).T
        np.savetxt('{}best-fit-wp_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),wp,header='s wp wperr')

        # distributions:
        UNITv,b = np.histogram(datac[:,1],range =(0,1500),bins=100)
        SHAMv = np.mean(xi1_ELG,axis=0)[4]
        bbins = (b[1:]+b[:-1])/2
        bbins = np.append(bbins,np.inf)
        UNITv = np.append(UNITv,len(datac))
        SHAMv = np.append(SHAMv,SHAMnum)
        np.savetxt('{}best-fit_Vpeak_hist_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),np.array([bbins,UNITv,SHAMv]).T,header='bins UNIT SHAM')

        # result report
        file = '{}Vzsmear_report_{}_{}.txt'.format(fileroot[:-10],gal,GC)
        f = open(file,'a')#;import pdb;pdb.set_trace()
        if date != '0729':
            f.write('python chi2 = {:.3f}, correspond to Vceil = {:.6}km/s \n'.format(res.dot(covR.dot(res)),np.mean(xi1_ELG,axis=0)[3]))
        pdf = SHAMv[:-1]/UNITv[:-1]
        f.write('z{}z{} PDF max: {} km/s \n'.format(zmin,zmax,(bbins[:-1])[pdf==max(pdf[~np.isnan(pdf)])]))      
        f.close()

    # plot the 2pcf results
    errbar = np.append(obscf['col4'],obscf['col5'])
    errbarsham = np.loadtxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC),usecols=(2,3))
    #print('mean Vceil:{:.3f}'.format(np.mean(true_array,axis=0)[2]))
    if rscale=='linear':
        Ccode = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[binmin:binmax]
    else:
        Ccode = np.loadtxt('{}best-fit_{}_{}.dat'.format(fileroot[:-10],gal,GC))[1:]
    pythoncode = np.loadtxt('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC))

    columns = []
    import re
    with open('{}best-fit_{}_{}-python.dat'.format(fileroot[:-10],gal,GC), 'r') as td:
        for line in td:
            if line[0] == '#':
                info = re.split(' +', line)
                columns.append(info)
    # plot the 2PCF multipoles   
    fig = plt.figure(figsize=(14,8))
    spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], wspace=0.2,hspace=0)
    ax = np.empty((2,2), dtype=type(plt.axes))
    for col,name,k in zip(['col2','col3'],['monopole','quadrupole'],range(2)):
        values=[np.zeros(nbins),obscf[col]]        
        err   = [np.ones(nbins),s**2*errbar[k*nbins:(k+1)*nbins]]
        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])
            #ax[j,k].plot(s,s**2*(xi[:,k]-values[j]),c='c',alpha=0.6,label='SHAM-python')
            #ax[j,k].errorbar(s+0.1,s**2*(Ccode[:,k+2]-values[j])/err[j],s**2*errbarsham[:,k]/err[j],c='m',alpha=0.6,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-npar)))
            ax[j,k].plot(s,s**2*(pythoncode[:,k]-values[j])/err[j],c='k',alpha=0.6,label='_hidden')#'SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-npar)))
            ax[j,k].fill_between(s,s**2*(pythoncode[:,k]-values[j])/err[j]-s**2*errbarsham[:,k]/err[j],s**2*(pythoncode[:,k]-values[j])/err[j]+s**2*errbarsham[:,k]/err[j],color='k',alpha=0.4,label='SHAMpython, $\chi^2$/dof={:.4}/{}'.format(float(columns[0][-1]),int(2*len(s)-npar)))
            ax[j,k].errorbar(s,s**2*(obscf[col]-values[j])/err[j],s**2*errbar[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs+{}'.format('jackknife'))
            ax[j,k].plot(s,s**2*(Ccode[:,k+2]-values[j])/err[j],c='m',alpha=0.6,label='_hidden')#'SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-npar)))
            ax[j,k].fill_between(s,s**2*(Ccode[:,k+2]-values[j])/err[j]-s**2*errbarsham[:,k]/err[j],s**2*(Ccode[:,k+2]-values[j])/err[j]+s**2*errbarsham[:,k]/err[j],color='m',alpha=0.4,label='SHAM, $\chi^2$/dof={:.4}/{}'.format(-2*a.get_best_fit()['log_likelihood'],int(2*len(s)-npar)))
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

    plt.savefig('{}cf_{}_bestfit_{}_{}_z{}z{}_{}-{}Mpch-1_{}.png'.format(fileroot[:-10],multipole,gal,GC,zmin,zmax,rmin,rmax,date))
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
        plt.ylim(0,1.05)
    elif gal == 'ELG':
        plt.ylim(0,0.3)
    plt.ylabel('prob of having a galaxy')
    plt.xlabel('Vpeak (km/s)')
    plt.savefig(fileroot[:-10]+'best_SHAM_PDF_hist_{}_{}.png'.format(gal,GC))
    plt.close()
    pdf = SHAMv[:-1]/UNITv[:-1]
    print('z{}z{} PDF max: {} km/s'.format(zmin,zmax,(bbins[:-1])[pdf==max(pdf[~np.isnan(pdf)])]))

    # plot the wp
    fig = plt.figure(figsize=(6,7))
    spec = gridspec.GridSpec(nrows=2,ncols=1, height_ratios=[4, 1], hspace=0.3)
    ax = np.empty((2,1), dtype=type(plt.axes))
    for k in range(1):
        values=[np.zeros_like(OBSwp),OBSwp]
        err   = [np.ones_like(OBSwp),errbarwp]

        for j in range(2):
            ax[j,k] = fig.add_subplot(spec[j,k])#;import pdb;pdb.set_trace()
            ax[j,k].errorbar(s,(OBSwp-values[j])/err[j],errbarwp/err[j],color='k', marker='o',ecolor='k',ls="none",label='obs 1$\sigma$ $\pi$40')
            ax[j,k].plot(s,(wp[:,1]-values[j])/err[j],color='b',label='SHAM $\pi$40')
            ax[j,k].fill_between(s,(wp[:,1]-values[j]-wp[:,2])/err[j],(wp[:,1]-values[j]+wp[:,2])/err[j],color='b',alpha=0.4,label='_hidden')
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

    plt.savefig('{}wp_bestfit_{}_{}_{}-{}Mpch-1_pi40_{}.png'.format(fileroot[:-10],gal,GC,rmin,rmax,date))
    plt.close()

    # Vsmear vs Delta v 
    # read SHAM best-fit
    stats = a.get_stats()    
    median = stats['marginals'][1]['median']
    diff = stats['marginals'][1]['1sigma'][1]-median
    # select repeats
    import fitsio
    catapath = '/global/homes/j/jiaxi/DESI_redshift_uncertainty/repeats_final-targets/'
    hdu = fitsio.read(catapath+'DESI_{}_redshift_uncertainty.fits.gz'.format(gal))
    tcomp = Table(hdu)
    sel = (tcomp['Z_TRUTH']<zmax)&(tcomp['Z_TRUTH']>zmin)
    sel &= tcomp['ZWARN'] == 0
    tcomp = tcomp[sel]
    # the fitting function
    if Vsmeartype == 'Gaussian':
        def func(x,a,sigma,mu):
            return a/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2*sigma**2))
        label = r'Gaussian fit $\sigma = {:.1f} \pm {:.1f}$'
        label1 = r'SHAM Gaussian fit $\sigma = {:.1f} \pm {:.1f}$'
        rescale = 1
    else:
        def func(x,a,w,p):
            return a/(1+((x-p)*2/w)**2)
        label   = r'Lorentzian fit w/(2$\sqrt{{2ln2}})$'+' = ${:.1f} \pm {:.1f}$'
        label1   = r'SHAM Lorentzian fit w/(2$\sqrt{{2ln2}})$'+' = ${:.1f} \pm {:.1f}$'
        rescale = 2*np.sqrt(2*np.log(2))

    for mode in ['multiple-obs', 'one-obs', 'inter-obs']:
        print('repetitive model:',mode)
        if mode == 'one-obs':
            modes = 'one obs - deepest obs'
            dv = (tcomp['Z'] - tcomp['Z_TRUTH'])*299792./(1+tcomp['Z_TRUTH'])
            zerr = tcomp['ZERR']*299792./(1+tcomp['Z_TRUTH'])
            A,B = np.unique(tcomp['TARGETID'],return_index=True)
            dv = dv[B]
        elif mode == 'inter-obs':
            modes = 'one obs - another obs'
            A = np.unique(tcomp['TARGETID'])
            tcomp['ZERR_TRUTH']  = np.zeros(len(tcomp))
            print(len(A))
            for az in A:
                tcomp[tcomp['TARGETID'] == az]['Z_TRUTH'] == tcomp[tcomp['TARGETID'] == az]['Z'][0]
                tcomp[tcomp['TARGETID'] == az]['ZERR_TRUTH'] == tcomp[tcomp['TARGETID'] == az]['ZERR'][0]
            dv = (tcomp['Z'] - tcomp['Z_TRUTH'])*299792./(1+tcomp['Z_TRUTH'])
            dv = dv[dv!=0]
        elif mode =='multiple-obs':
            modes = 'all obs - deepest obs'
            dv = (tcomp['Z'] - tcomp['Z_TRUTH'])*299792./(1+tcomp['Z_TRUTH'])
            
        # make histograms
        BIN = (vbins[1:]+vbins[:-1])/2
        dens,BINS,plot = plt.hist(dv, bins=vbins, histtype='step',label='repeat obs')

        # fitting & plot
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func,BIN,dens)    
        plt.plot(BIN, func(BIN,*popt),c='k',label=label.format(popt[1]/rescale,np.sqrt(np.diag(pcov))[1]/rescale))
        # plot the SHAM best-fit
        plt.plot(BIN, func(BIN,popt[0],median,popt[2]),c='green',label=label1.format(median/rescale,diff/rescale))        
        plt.title('repeat observation with dv = {}'.format(modes))
        plt.ylim(0,1.5*max(dens))
        plt.legend(loc=0)
        plt.savefig('{}best-fit_vsmear_{}_z{}z{}_{}.png'.format(fileroot[:-10],gal,zmin,zmax,mode))
        plt.close()


