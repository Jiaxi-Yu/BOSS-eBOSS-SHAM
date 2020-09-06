import matplotlib 
matplotlib.use('agg')
import time
init = time.time()
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack
from astropy.table import Table
from astropy.io import fits
from Corrfunc.theory.DDsmu import DDsmu
import os
from covmatrix import covmatrix
from obs import obs
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import repeat
import glob
import matplotlib.gridspec as gridspec 
import sys
import pymultinest

# variables
date2    = '0810'
nseed    = 2
rscale   = 'linear' # 'log'
multipole= 'quad' # 'mono','quad','hexa'
var      = 'Vpeak'  #'Vmax' 'Vpeak'
Om       = 0.31
boxsize  = 1000
rmin     = 5
rmax     = 25
nthread  = 64
autocorr = 1
mu_max   = 1
nmu      = 120
autocorr = 1
PDFmax   = 3.5
PDFmin   = 1.5
home      = '/global/cscratch1/sd/jiaxi/master/'
scatrange = [2.573,2.588,2.526,2.540] 
scatnum = 50

# data for ELG
LRGnum2   = int(2.93e5)
zmin     = 0.6
zmax     = 1.1
z2 = 0.8594
halofile2 = home+'catalog/UNIT_hlist_0.53780.fits.gz' 

# cosmological parameters
Ode = 1-Om
# generate separation bins
if rscale=='linear':
    bins  = np.arange(rmin,rmax+1,1)
    nbins = len(bins)-1
    binmin = rmin
    binmax = rmax

# generate mu bins   
s = (bins[:-1]+bins[1:])/2
mubins = np.linspace(0,mu_max,nmu+1)
mu = (mubins[:-1]+mubins[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))

# Analytical RR calculation
RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)	
rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)
print('the analytical random pair counts are ready.')

# create the halo catalogue and plot their 2pcf
print('reading the halo catalogue for creating the galaxy catalogue...')
# make sure len(data) is even
halo = fits.open(halofile2)
if len(halo[1].data)%2==1:
    data = halo[1].data[:-1]
else:
    data = halo[1].data
halo.close()
print('2. selecting only the necessary variables...')
datac2 = np.zeros((len(data),5))
for i,key in enumerate(['X','Y','Z','VZ',var]):
    datac2[:,i] = np.copy(data[key])
#V = np.copy(data[var]).astype('float32')
datac2 = datac2.astype('float32')
data = np.zeros(0)

# HAM application
def sham_tpcf(dat,uni,uni1,sigM,sigV,Mtrun):
    x00,x20,x40,n20,n22=sham_cal(dat,datac2,z2,LRGnum2,uni,sigM,sigV,Mtrun)
    x01,x21,x41,n21,n23=sham_cal(dat,datac2,z2,LRGnum2,uni1,sigM,sigV,Mtrun)
    return [(x00+x01)/2,(x20+x21)/2,(x40+x41)/2,(n20+n21)/2,(n22+n23)/2]

def sham_cal(dat,DATAC,z,LRGnum,uniform,sigma_high,sigma,v_high):
    half = int(len(DATAC)/2)
    datav = DATAC[:,-1]*(1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))) #0.5s
    LRGscat = (DATAC[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:LRGnum]]
    n,bins0=np.histogram(np.log10(LRGscat[:,-1]),bins=50,range=(PDFmin,PDFmax))
    plt.scatter(LRGscat[:,-1],(datav[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:LRGnum]],c='r',alpha=0.4,s=1)
    plt.xlabel('Vpeak')
    plt.ylabel('scattered Vpeak')
    plt.savefig('vpeak_vs_scat_{:.3}.png'.format(uniform[0]))
    plt.close()
    if dat==1:
        n01,bins01=np.histogram(np.log10((datav[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]),bins=scatnum,range=(scatrange[dat-1],scatrange[dat]))
    else:
        n01,bins01=np.histogram(np.log10((datav[datav<v_high])[argpartition(-datav[datav<v_high],LRGnum)[:(LRGnum)]]),bins=scatnum,range=(scatrange[dat],scatrange[dat+1]))

    # transfer to the redshift space
    scathalf = int(LRGnum/2)
    H = 100*np.sqrt(Om*(1+z)**3+Ode)
    z_redshift  = (LRGscat[:,2]+(LRGscat[:,3]+append(sigma*sqrt(-2*log(uniform[:scathalf]))*cos(2*pi*uniform[-scathalf:]),sigma*sqrt(-2*log(uniform[:scathalf]))*sin(2*pi*uniform[-scathalf:])))*(1+z)/H)
    z_redshift %=boxsize
    # count the galaxy pairs and normalise them
    DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,0],LRGscat[:,1],z_redshift,periodic=True, verbose=True,boxsize=boxsize)
    # calculate the 2pcf and the multipoles
    mono = (DD_counts['npairs'].reshape(nbins,nmu)/(LRGnum**2)/rr-1)
    quad = mono * 2.5 * (3 * mu**2 - 1)
    hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)
    # use trapz to integrate over mu
    xi0_single = np.trapz(mono, dx=1./nmu, axis=-1)
    xi2_single = np.trapz(quad, dx=1./nmu, axis=-1)
    xi4_single = np.trapz(hexa, dx=1./nmu, axis=-1)
    print('calculation finish')
    return [xi0_single,xi2_single,xi4_single,n,n01]

# read the posterior file
fileroot1 = 'MCMCout/3-param_'+date2+'/ELG_SGC/multinest_'
parameters1 = ["sigma","Vsmear","Vceil"]
npar1 = len(parameters1)
a1 = pymultinest.Analyzer(npar1, outputfiles_basename = fileroot1)

fileroot2 = 'MCMCout/3-param_'+date2+'/ELG_NGC/multinest_'
parameters2 = ["sigma","Vsmear","Vceil"]
npar2 = len(parameters2)
a2 = pymultinest.Analyzer(npar2, outputfiles_basename = fileroot2)

# plot the best-fit for ELGs
# generate uniform random numbers
print('generating uniform random number arrays...')
uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac2)).astype('float32') for x in range(nseed)] 
uniform_randoms1 = [np.random.RandomState(seed=1050*x+1).rand(len(datac2)).astype('float32') for x in range(nseed)] 


with Pool(processes = nseed) as p:
    xi_ELG = p.starmap(sham_tpcf,zip(repeat(1),uniform_randoms,uniform_randoms1,repeat(np.float32(a1.get_best_fit()['parameters'][0])),repeat(np.float32(a1.get_best_fit()['parameters'][1])),repeat(np.float32(a1.get_best_fit()['parameters'][2])))) 

with Pool(processes = nseed) as p:
    xi1_ELG = p.starmap(sham_tpcf,zip(repeat(2),uniform_randoms,uniform_randoms1,repeat(np.float32(a2.get_best_fit()['parameters'][0])),repeat(np.float32(a2.get_best_fit()['parameters'][1])),repeat(np.float32(a2.get_best_fit()['parameters'][2]))))  



# UNIT [PDF,CDF]
n2,bins2=np.histogram(np.log10(datac2[:,-1]),bins=50,range=(PDFmin,PDFmax))
N_UNIT = [n2/np.sum(n2),np.array([np.sum(n2[:x])/np.sum(n2) for x in range(50)])]

# log(Vpeak_selected)
bins0=(np.linspace(PDFmin,PDFmax,51)[1:]+np.linspace(PDFmin,PDFmax,51)[:-1])/2
n1list = [xi_ELG[x][3] for x in range(nseed)]
n1array = np.array(n1list).T #NGC SHAM
n2list = [xi1_ELG[x][3] for x in range(nseed)]
n2array = np.array(n2list).T #SGC SHAM

# log(Vpeak_scat)
n10list = [xi_ELG[x][4] for x in range(nseed)]
n10array   = np.array(n10list).T/LRGnum2
n20list = [xi1_ELG[x][4] for x in range(nseed)]
n20array   = np.array(n20list).T/LRGnum2
'''
mode = 'PDF' # log(Vpeak_all)log(Vpeak_selected),log(Vpeak_scat) vs log(M*)
#mode = 'PDF-CDF' # log(Vpeak_all)log(Vpeak_selected) PDF vs CDF
for GC,array,array_scat in zip(['NGC','SGC'],[n1array,n2array],[n10array,n20array]):
    file = '/global/cscratch1/sd/jiaxi/master/catalog/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits'
    hdu = fits.open(file)
    data = hdu[1].data['fast_lmass']
    N_M,binm = np.histogram(data,bins=scatnum)
    N_Mtot = [N_M/np.sum(N_M) ,np.array([np.sum(N_M[:x])/np.sum(N_M) for x in range(scatnum)])] #clustering
    if mode == 'PDF-CDF':
        fig =plt.figure(figsize=(10,17))
        for i,types in enumerate(['PDF','CDF']):
            Narray = [array,np.array([np.sum(array[:x],axis=0) for x in range(50)])] 
            for j,prob,binmid,label,unit in zip(range(3),[Narray[i],N_Mtot[i],N_UNIT[i]],[bins0,(binm[1:]+binm[:-1])/2,bins0],['ELG SHAM','ELG clustering','UNIT'],['$log_{10}(V_{peak})$','$log_{10}(M_*)$','$log_{10}(V_{peak})$']):
                ax = plt.subplot2grid((3,2),(j,i))
                #print(binmid.shape,prob.shape)
                if j==0:
                    if i==0:
                        ax.errorbar(binmid,np.mean(prob,axis=-1)/np.sum(np.mean(prob,axis=-1)),yerr = np.std(prob,axis=-1)/np.sum(np.mean(prob,axis=-1)),color='m',alpha=0.7,ecolor='m',ds='steps-mid')
                    else:
                        ax.errorbar(binmid,np.mean(prob,axis=-1)/(np.mean(prob,axis=-1)).max(),yerr = np.std(prob,axis=-1)/(np.mean(prob,axis=-1)).max(),color='m',alpha=0.7,ecolor='m',ds='steps-mid')
                else:
                    ax.errorbar(binmid,prob,yerr = np.zeros_like(prob),color='m',alpha=0.7,ecolor='m',ds='steps-mid')
                plt.ylabel('probability')
                plt.xlabel(unit)
                plt.title('{} {} {} in {} '.format(label,unit,types,GC))
                if i==0:
                    ax.set_yscale('log')
                    ax.set_ylim(1e-8,1)
                    if j!=1:
                        ax.set_xlim(PDFmin,PDFmax)
        plt.savefig('PDF&CDF_ELG_'+GC+'.png',bbox_tight=True)
        plt.close()
    else:
        print('second')
        fig = plt.figure(figsize=(18,5))
        ax  = plt.subplot2grid((1,3),(0,0))
        ax.errorbar(bins0,np.mean(array,axis=-1)/np.sum(np.mean(array,axis=-1)),yerr = np.std(array,axis=-1)/np.sum(np.mean(array,axis=-1)),color='m',alpha=0.7,ecolor='m',ds='steps-mid',label = 'SHAM log($V_{peak}$)')
        ax.errorbar(bins0,N_UNIT[0],yerr = np.zeros_like(N_UNIT[0]),color='k',alpha=0.7,ecolor='m',ds='steps-mid',label='UNIT log($V_{peak}$)')
        ax.set_xlim(PDFmin,PDFmax)
        plt.ylabel('probability')
        plt.title('simulated probability distributions in {} '.format(GC))
        ax.set_yscale('log')
        ax.set_ylim(1e-8,1)
        ax  = plt.subplot2grid((1,3),(0,1))
        if GC=='NGC':
            scatbin = (np.linspace(scatrange[0],scatrange[1],scatnum+1)[:-1]+np.linspace(scatrange[0],scatrange[1],scatnum+1)[1:])/2
        else:
            scatbin = (np.linspace(scatrange[2],scatrange[3],scatnum+1)[:-1]+np.linspace(scatrange[2],scatrange[3],scatnum+1)[1:])/2
        ax.errorbar(scatbin,np.mean(array_scat,axis=-1),yerr = np.std(array_scat,axis=-1),color='c',alpha=0.7,ecolor='m',ds='steps-mid',label = 'SHAM log($V_{peak}^{scat}$)')
            
        plt.ylabel('probability')
        plt.title('scattered Vpeak in {} '.format(GC))
        ax.set_yscale('log')
        ax.set_ylim(1e-8,1)
        ax  = plt.subplot2grid((1,3),(0,2))
        ax.errorbar((binm[1:]+binm[:-1])/2,N_Mtot[0],yerr = np.zeros_like(N_Mtot[0]),color='k',alpha=0.7,ecolor='m',ds='steps-mid',label='log(M*)')
        plt.ylabel('probability')
        plt.title('realistic probability distributions in {} '.format(GC))
        ax.set_yscale('log')
        ax.set_ylim(1e-8,1)
        plt.savefig('PDF_ELG_'+GC+'.png',bbox_tight=True)
        plt.close()
'''        
# calculate CDF for SHAM PDFs in different seeds
n10CDF = [j for j in range(nseed)]
n20CDF = [j for j in range(nseed)]
for j in range(nseed):
    n10CDF[j]   = np.array([np.sum(n10array[:,j][-x-1:]) for x in range(scatnum)])
    n20CDF[j]   = np.array([np.sum(n20array[:,j][-x-1:]) for x in range(scatnum)])

# match the stellar mass and the scattered Vpeak.
# initialisation: the 1st element: scattered Vpeak
CDF2CDF1 = [(np.linspace(scatrange[0],scatrange[1],scatnum+1)[:-1]+np.linspace(scatrange[0],scatrange[1],scatnum+1)[1:])/2,np.zeros(scatnum),np.zeros(scatnum)]
CDF2CDF2 = [(np.linspace(scatrange[2],scatrange[3],scatnum+1)[:-1]+np.linspace(scatrange[2],scatrange[3],scatnum+1)[1:])/2,np.zeros(scatnum),np.zeros(scatnum)]

for GC,CDF,CDF2CDF in zip(['NGC','SGC'],[n10CDF,n20CDF],[CDF2CDF1,CDF2CDF2]):
    file = '/global/cscratch1/sd/jiaxi/master/catalog/eBOSS_ELG_clustering_'+GC+'_v7.dat.fits'
    hdu = fits.open(file)
    data = hdu[1].data['fast_lmass']
    data = data[np.argsort(-data)] 
    hdu.close()
    # the 3rd element: the std of the CDF
    CDF2CDF[2] = np.std(CDF,axis=0)
    # the 2nd element: the corresponding M* for the scattered Vpeak CDF
    for i,ind in enumerate(np.ceil(np.mean(CDF,axis=0)*len(data))):
        if ind ==0:
            CDF2CDF[1][i] = data[0]
        else:
            CDF2CDF[1][i] = data[int(ind)-1]
    plt.errorbar(CDF2CDF[0][(np.mean(CDF,axis=0)!=0)&(np.mean(CDF,axis=0)!=1)],CDF2CDF[1][(np.mean(CDF,axis=0)!=0)&(np.mean(CDF,axis=0)!=1)],CDF2CDF[2][(np.mean(CDF,axis=0)!=0)&(np.mean(CDF,axis=0)!=1)],color='k',alpha=0.7,ecolor='k')
    plt.title('scattered Vpeak - M* relation of ELG in {}'.format(GC))
    plt.xlabel('log(scattered Vpeak)')
    plt.ylabel('log(M*)')
    plt.savefig('ELG_'+GC+'_scattered_Vpeak-stellar_mass_relation.png')
    plt.close()
# plot the Vpeak-M* relation
