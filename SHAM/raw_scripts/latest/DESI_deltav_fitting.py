import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
from astropy.io import fits

import evalSR
import importlib
importlib.reload(evalSR)

from scipy.optimize import curve_fit
from scipy.stats import cauchy

datadir = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
task = sys.argv[1]
# "plot" 'random_catalogue', 'lorentzian_random', 'voigt_random'
TYPE = sys.argv[2]
# 'LRG', 'ELG', 'QSO', 'BGS'
dz,z,zrsel,ggzsel = {'LRG':[],'ELG':[],'QSO':[],'BGS':[]},{'LRG':[],'ELG':[],'QSO':[],'BGS':[]},{'LRG':[],'ELG':[],'QSO':[],'BGS':[]},{'LRG':[],'ELG':[],'QSO':[],'BGS':[]}
#-- fit a Gaussian
def gaussian(x,a,sigma,mu):
    return a/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2*sigma**2))
def lorentzian(x,a,w,p):
    return a/(1+((x-p)*2/w)**2)
def pvoigt(x,a,sigma,mu,a1):
    w = sigma*2*np.sqrt(2*np.log(2))
    return gaussian(x,a,sigma,mu)+lorentzian(x,a1,w,mu)
    #return a1*np.convolve(gaussian(x,a,sigma,mu),lorentzian(x,a/np.pi*w,w,mu),mode='same')
def sampling(datalen):
    Nsample = datalen*5
    simu = []
    samples = []
    bins = np.arange(-500,501,5)
    BIN = (bins[1:]+bins[:-1])/2
    #import pdb;pdb.set_trace()
    norm = len(dv)
    dens,BINS,plot = plt.hist(dv, bins=bins, histtype='step')
    cQ = 1/len(BIN)*10
    #popt2, pcov21 = curve_fit(pvoigt,BIN,dens/norm,bounds=(np.array([0,0,-np.inf,0]),np.array([np.inf,1000,np.inf,np.inf])))
    #import pdb;pdb.set_trace()

    for i in range(Nsample):
        random = np.random.uniform(low=-maxdv,high=maxdv)
        value = np.interp(random,BIN,dens/norm)
        #value = 10*pvoigt(random,popt2[3]*np.pi*popt2[1],popt2[1],popt2[2],popt2[3])
        #import pdb;pdb.set_trace()
        sample = np.random.uniform(0,cQ)
        if sample<value:
            simu.append(random)
        samples.append(random)
        if len(simu)==datalen:
            break
    return np.array(simu)

if TYPE == 'LRG':
    zmin=0.32;zmax=0.6
    maxdv = 200
    bins = np.arange(-maxdv,maxdv+1,5)
elif TYPE == 'ELG':
    zmin=0.6;zmax=1.6
    maxdv = 100
    bins = np.arange(-maxdv,maxdv+1,1)
elif TYPE == 'QSO':
    zmin=0.6;zmax=2.1
    maxdv = 400
    bins = np.arange(-maxdv,maxdv+1,5)
elif TYPE =='BGS':
    zmin=0.01;zmax=0.5
    maxdv = 150
    bins = np.arange(-maxdv,maxdv+1,5)

zrange = 'z{}z{}'.format(zmin,zmax) #'z0.4z1.0' 'z0.32z0.6' 'z0.6z0.8'
print('delta v: reading the catalogue')
if not os.path.exists(datadir+'DESI_{}_redshift_uncertainty.fits.gz'.format(TYPE)):
    tf = evalSR.add_truth(TYPE,release='cascades',version='3.1',bdir='/global/cfs/cdirs/desi/survey/catalogs/SV1/redshift_comps')
    # zrsel for the redshift-range selection, ggsel for the redshift-range and removing-catastrophic selection
    if TYPE =='QSO':
        dz[TYPE],z[TYPE],ggzsel[TYPE] = evalSR.repeatvsdchi2(tf,TYPE+'lz')
    else:
        dz[TYPE],z[TYPE],ggzsel[TYPE] = evalSR.repeatvsdchi2(tf,TYPE)
else:
    hdu = fits.open(datadir+'DESI_{}_redshift_uncertainty.fits.gz'.format(TYPE))
    tcomp = hdu[1].data
    hdu.close()
    sel = (tcomp['Z_TRUTH']<zmax)&(tcomp['Z_TRUTH']>zmin)
    sel &= tcomp['ZWARN'] == 0
    tcomp = tcomp[sel]
    ztrue = tcomp['Z_TRUTH']*1
    dv = (tcomp['Z'] - tcomp['Z_TRUTH'])*299792./(1+tcomp['Z_TRUTH'])
    zerr = tcomp['ZERR']*299792./(1+tcomp['Z_TRUTH'])

    #ztrue,dv,zerr = np.loadtxt('{}_deltav_{}_new.dat'.format(TYPE,zrange),unpack=True)

if task == 'plot':
    print('delta v: fitting')
    for jseq in range(2):
        fig,ax    = plt.subplots(1,1,figsize=(9,6))
        BIN = (bins[1:]+bins[:-1])/2
        #import pdb;pdb.set_trace()
        norm = len(dv)
        dens,BINS,plot = plt.hist(dv, bins=bins, histtype='step')
        
        popt, pcov = curve_fit(gaussian,BIN,dens)    
        print('Gaussian finished')
        popt1, pcov1 = curve_fit(lorentzian,BIN,dens)
        print('Lorentzian finished')
        popt2, pcov2 = curve_fit(pvoigt,BIN,dens,bounds=(np.array([0,0,-np.inf,0]),np.array([np.inf,1000,np.inf,np.inf])))
        print('Voigt finished')
        
        """
        # manual voigt realisation
        from lmfit.models import PseudoVoigtModel
        mod = PseudoVoigtModel()
        pars = mod.guess(dens, x=BIN)
        pars['fraction'].set(value=0.1, min=0, max=0.4)
        out = mod.fit(dens, pars, x=BIN)
        #print(out.fit_report(min_correl=0.25))
        plt.plot(BIN, out.best_fit,c='green',label='PseudoVoigt')# $p_0 = {:.1f} \pm {:.1f}$, '.format(popt1[2],np.sqrt(np.diag(pcov1))[2])+r'w/(2$\sqrt{2ln2})$'+' = ${:.1f} \pm {:.1f}$'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2))))
        """
        
        plt.plot(BIN, gaussian(BIN,*popt),c='r',label=r'Gaussian $\sigma = {:.1f} \pm {:.1f}, \chi^2/dof={:.1}/{}$'.format(popt[1],np.sqrt(np.diag(pcov))[1],sum((dens- gaussian(BIN,*popt))**2),len(dens)-3)) # $\mu = {:.1f} \pm {:.1f}, popt[2],np.sqrt(np.diag(pcov))[2], 
        plt.plot(BIN, lorentzian(BIN,*popt1),c='k',label='Lorentzian w/(2$\sqrt{2ln2})$'+' = ${:.1f} \pm {:.1f}, \chi^2/dof={:.1}/{}$'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2)),sum((dens- lorentzian(BIN,*popt1))**2),len(dens)-3)) #$p_0 = {:.1f} \pm {:.1f}$, '.format(popt1[2],np.sqrt(np.diag(pcov1))[2])+r', 
        #plt.plot(BIN, pvoigt(BIN,*popt2),c='g',label=r'PseudoVoigt Gaussian $\sigma = {:.1f} \pm {:.1f}, \chi^2/dof={:.1}/{}$'.format(popt2[1],np.sqrt(np.diag(pcov2))[1],sum((dens- pvoigt(BIN,*popt2))**2),len(dens)-3)) # $\mu = {:.1f} \pm {:.1f}, popt2[2],np.sqrt(np.diag(pcov2))[2], 
        outliern = len(dv[(dv>-1000)&(dv<-maxdv)])
        outlierp = len(dv[(dv<1000)&(dv>maxdv)])
        plt.scatter(-maxdv,outliern,c='b',label='outliers')
        plt.scatter(maxdv,outlierp,c='b')
        plt.title(TYPE+' dv histogram ,stdev = {:.1f} km/s'.format(np.std(dv)))
        ax.set_xlim(-maxdv-5,maxdv+5)
        if jseq ==0:
            log = 'lin'
            plt.ylim(0,1.8*max(dens))
        else:
            log = 'log'
            plt.ylim(1e-1,30*max(dens))
            plt.yscale('log')
        ax.set_xlabel('$\Delta$ v (km/s)')
        ax.set_ylabel('counts')
        ax.grid(True)
        plt.legend(loc=2)
        plt.savefig('{}_deltav_hist_std{:.1f}_{}_maxdv{}-{}_new.png'.format(TYPE,np.std(dv),zrange,maxdv,log))
        plt.close()
    print('{} in {} has {} pairs, fitting results are:'.format(TYPE,zrange,len(dv)))
    print('mu = {:.1f},sigma = {:.1f},chi2 = {:.1}'.format(popt[2],popt[1],sum((dens-gaussian(BIN,*popt))**2)))
    #print('abs<100 chi2 = {:.1}, abs>100 chi2 = {:.1}'.format(sum((dens[abs(BIN)<100]-gaussian(BIN,*popt)[abs(BIN)<100])**2),sum((dens[abs(BIN)>=100]-gaussian(BIN,*popt)[abs(BIN)>=100])**2)))
    print('lorentizan p0 = {:.1f}, sigma = {:.1f}, chi2 = {:.2}'.format(popt1[2],popt1[1]/np.sqrt(2*np.log(2))/2,sum((dens-lorentzian(BIN,*popt1))**2)))
    print('abs<100 chi2 = {:.1}, abs>100 chi2 = {:.1f}'.format(sum((dens[abs(BIN)<100]-lorentzian(BIN,*popt1)[abs(BIN)<100])**2),sum((dens[abs(BIN)>=100]-lorentzian(BIN,*popt1)[abs(BIN)>=100])**2)))
    print('voigt mu = {:.1f}, sigma = {:.1f}, chi2={:.2}'.format(popt2[2],popt2[1],sum((dens-pvoigt(BIN,*popt2))**2)))
    print('abs<100 chi2 = {:.1}, abs>100 chi2 = {:.1f}'.format(sum((dens[abs(BIN)<100]-pvoigt(BIN,*popt2)[abs(BIN)<100])**2),sum((dens[abs(BIN)>=100]-pvoigt(BIN,*popt2)[abs(BIN)>=100])**2)))
    print('stdev = ',np.std(dv))

    #-- fit a Gaussian for zerr/Delta v
    ratios = np.linspace(-10,10,101)
    ratio = (ratios[1:]+ratios[:-1])/2
    ratiodens,ratiobin = np.histogram(dv/zerr,ratios)
    #ratiodens = ratiodens/sum(ratiodens)

    popt3, pcov3 = curve_fit(gaussian,ratio,ratiodens)
    res2 = gaussian(ratio,*popt3)-ratiodens

    popt4, pcov4 = curve_fit(lorentzian,ratio,ratiodens)
    res3 = lorentzian(ratio,*popt4)-ratiodens
    #import pdb;pdb.set_trace()
    plt.figure(figsize=(8,6))
    plt.scatter(ratio,ratiodens)
    plt.scatter(ratio,ratiodens,color='k')
    plt.plot(ratio, gaussian(ratio,*popt3), label=r'Gaussian $\sigma = {0:.1f}\pm{{{1:.2f}}}$, $\chi^2$ /dof = {2:.1}/{3:}'.format(popt3[1],np.sqrt(np.diag(pcov3))[1],sum(res2**2),len(res2)))
    plt.plot(ratio, lorentzian(ratio,*popt4), label = 'Lorentzian w/(2$\sqrt{2ln2})$'+' = ${:.1f} \pm {:.1f}, \chi^2/dof={:.1}/{}$'.format(popt4[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov4))[1]/2/np.sqrt(2*np.log(2)),sum(res3**2),len(res3)))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('counts')
    plt.legend(loc=1)
    plt.ylim(0,max(ratiodens)*1.3)
    plt.title(TYPE+' dv/ZERR at {}<z<{}'.format(zmin,zmax))

    plt.savefig('{}_dvzerr_hist_{}_maxdv{}.png'.format(TYPE,zrange,maxdv))
    plt.close()
    #import pdb;pdb.set_trace()
    dvzerrsel = (abs(dv)<maxdv*2)&(zerr<maxdv/5*2)
    plt.hexbin(dv[dvzerrsel],zerr[dvzerrsel],bins='log')
    plt.xlabel('dv')
    plt.ylabel('zerr')
    plt.savefig('{}_hexbin_dvzerr_{}.png'.format(TYPE,zrange))
    plt.close()


elif task == 'random_catalogue':
    # generate the random numbers for cosmosim-0.74
    sigma = popt[1]
    gamma = popt1[1]/2
    stdev = np.std(dv)
    # Vsmear random numbers
    z = 0.74
    boxsize = 1000
    Om = 0.31
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)
    #sourcefile = '/global/project/projectdirs/desi/cosmosim/UNIT-BAO-RSD-challenge/Stage2Recon/UNITSIM/LRG/LRG-wpmax-v3-snap103-redshift{}_dens0.dat'.format(z)
    sourcefile = '/global/cscratch1/sd/jiaxi/SHAM/GLAM_0.74/'
    data = np.loadtxt(sourcefile+'mocks_rsd/0.dat')
    #import pdb;pdb.set_trace()

    for index in range(40): #40
        
        random_lorentzian = cauchy.rvs(loc=0, scale=gamma, size=len(data))
        while len(random_lorentzian[abs(random_lorentzian)>1000])>0:
            random_lorentzian[abs(random_lorentzian)>1000] = cauchy.rvs(loc=0, scale=gamma, size = len(random_lorentzian[abs(random_lorentzian)>1000]))        
        
        random_lorentzian_trunc = cauchy.rvs(loc=0, scale=gamma, size=len(data))
        while len(random_lorentzian_trunc[(abs(random_lorentzian_trunc)>maxdv)])>0:
            random_lorentzian_trunc[abs(random_lorentzian_trunc)>maxdv] = cauchy.rvs(loc=0, scale=gamma, size = len(random_lorentzian_trunc[(abs(random_lorentzian_trunc)>maxdv)]))        
                
        random_gaussian   = np.random.normal(loc=0,scale = sigma,size = len(data))
        random_stdev      = np.random.normal(loc=0,scale = stdev,size = len(data))
        
        random_data      = sampling(len(data))
        vsmear = []
        
        if index == 0:
            fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(12, 5),sharey=True)
            den,BINS = np.histogram(dv, bins=bins)#, histtype='step',label='data')
            ax[0].plot(BIN,den/norm,'k',label='data')
            dens,BINS = np.histogram(random_data, bins=bins)#, histtype='step',label='data')
            ax[0].plot(BIN,dens/len(data),'r--',label='sampled data')
            # Lorentzian related random array
            dens,BINS = np.histogram(random_lorentzian, bins=bins)#, histtype='step',label='lorentzian')
            ax[0].plot(BIN,dens/len(random_lorentzian),'b',label='Lorentzian')

            # Gaussian related random array
            ax[1].plot(BIN,den/norm,'k',label='data')
            dens,BINS = np.histogram(random_lorentzian_trunc, bins=bins)#, histtype='step',label='lorentzian')
            ax[1].plot(BIN,dens/len(random_lorentzian_trunc),'b',label='truncated Lorentzian')            
            dens,BINS = np.histogram(random_gaussian, bins=bins)#, histtype='step',label='lorentzian')
            ax[1].plot(BIN,dens/len(random_gaussian),'r',label='Gaussian')
            dens,BINS = np.histogram(random_stdev, bins=bins)#, histtype='step',label='lorentzian')
            ax[1].plot(BIN,dens/len(random_stdev),'orange',label='Gaussian(stdev)')

            ax[0].set_ylabel('normalised counts')
            for i in range(2):
                ax[i].set_xlim(-maxdv-5,maxdv+5)
                ax[i].set_xlabel('$\Delta$ v (km/s)')
                ax[i].set_ylim(0,1.8*max(den)/norm)
                ax[i].grid(True)
                ax[i].legend(loc=2)
            plt.savefig('{}_distribution_{}_maxdv{}.png'.format(TYPE,zrange,maxdv))
            plt.close()
        else:
            print('generating Vsmeared peculier velocities:',index)

            # save the random arrays
            """
            for k,randoms in enumerate([random_lorentzian,random_gaussian,random_stdev,random_lorentzian_trunc,random_data]):
                vsmear.append((data[:,-1]+(randoms*(1+z)/H)%boxsize)%boxsize)
            np.savetxt(sourcefile+'mock0_smear/mock0_smear{}.dat'.format(index),np.hstack((data,np.array(vsmear).T)))
            
            ## add the data-like distribution later
            vsmear = np.loadtxt(sourcefile+'mock0_smear/mock0_smear{}.dat'.format(index))
            vsmear_data = (data[:,-1]+(random_data*(1+z)/H)%boxsize)%boxsize
            np.savetxt(sourcefile+'mock0_smear/mock0_smear{}.dat'.format(index),np.hstack((vsmear,vsmear_data.reshape(len(data),1))))
            """
elif task == 'lorentzian_random':
    # test: histogram of the random samples agrees with the observations
    from scipy.stats import cauchy
    fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(12, 10))
    for i in range(2):
        if i==0:
            rand = cauchy.rvs(loc=0, scale=popt1[1]/2, size=int(5e5))#int(len(dv)))
            numS,binS = np.histogram(rand,bins =bins)#,density=True)
            label = 'simulated without extra samples'
        else:
            rand = cauchy.rvs(loc=0, scale=popt1[1]/2, size=int(6e5))#int(len(dv)*1.1))
            numS,binS = np.histogram(rand,bins =bins)#,density=True)
            label = 'simulated with extra samples'

        for j in range(2):
            ax[i,j].plot(BIN,numS/len(rand),'r--',label=label)
            #dens,BINS,plot = ax[i,j].hist(dv, bins=bins, histtype='step',label='data')
            dens,BINS = np.histogram(dv, bins=bins)#, histtype='step',label='data')
            ax[i,j].plot(BIN,dens/norm,'k',label='data')
            ax[i,j].plot(BIN,lorentzian(BIN,popt1[0]/norm,popt1[1],popt1[2]),label='analitical')
            ax[i,j].set_xlabel('$\Delta$ v')
            ax[i,j].set_ylabel('counts')
            if j==0:
                ax[i,j].legend(loc=0,fontsize=10)
                ax[i,j].set_ylim(0,1.8*max(dens)/norm)
            if j ==1:
                ax[i,j].set_yscale('log')
                ax[i,j].set_ylim(10/norm,1.8*max(dens)/norm)

    plt.savefig('lorentzian_distribution.png')
    plt.close()

elif task == 'voigt_random':
    # test pesudo-voigt distribution: have difficulty in normalisation
    # len(dv) is the normalisation of the standard voigt
    Nsample = int(5.5e5)
    x = []
    simu = []
    samples = []
    cQ = 1/len(BIN)*5
    #popt2, pcov21 = curve_fit(pvoigt,BIN,dens/norm,bounds=(np.array([0,0,-np.inf,0]),np.array([np.inf,1000,np.inf,np.inf])))
    #import pdb;pdb.set_trace()

    for i in range(Nsample):
        random = np.random.uniform(low=-maxdv,high=maxdv)
        value = np.interp(random,BIN,dens/norm)
        #value = 10*pvoigt(random,popt2[3]*np.pi*popt2[1],popt2[1],popt2[2],popt2[3])
        #import pdb;pdb.set_trace()
        sample = np.random.uniform(0,cQ)
        if sample<value:
            simu.append(random)
        samples.append(random)
        #if len(simu)==len(dv):
        #    break
    print('sampled {} points, get {} valid ones'.format(Nsample,len(simu)))
    numS,binS = np.histogram(simu,bins=bins)
    plt.plot(BIN,numS/len(simu),label='sampled distribution')
    numS,binS = np.histogram(simu[:40000],bins=bins)
    plt.plot(BIN,numS/40000,label='sampled distribution: downsampled')

    #numA,binA = np.histogram(samples,bins=bins)

    #print(numA)
    plt.plot(BIN,cQ*np.ones_like(numS),'r',label='sampled uniform')
    #plt.plot(testbin,1/norm*pvoigt(testbin,popt2[3]*np.pi*popt2[1],popt2[1],popt2[2],popt2[3]),color='orange',label='analytical')
    plt.plot(BIN,dens/norm,'k',label='data')
    plt.xlabel('$\Delta$ v')
    plt.ylabel('counts')
    plt.legend(loc=0)
    plt.yscale('log')
    plt.savefig('sampling.png')
    plt.close()
else:
    pass