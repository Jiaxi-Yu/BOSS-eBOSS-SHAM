#!/usr/bin/env python3
import matplotlib 
matplotlib.use('agg')

import os
if not os.path.exists('param_evolution/real_space/SHAM_amplitude_z0.8z1.0.dat'):
    print('generate 2pcf')
    import time
    init = time.time()
    import numpy as np
    from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
    from astropy.table import Table
    from astropy.io import fits
    from Corrfunc.theory.DDsmu import DDsmu
    from Corrfunc.theory.wp import wp
    import warnings
    import matplotlib.pyplot as plt
    from multiprocessing import Pool 
    from itertools import repeat
    import glob
    import matplotlib.gridspec as gridspec
    from getdist import plots, MCSamples, loadMCSamples
    import sys
    import corner
    import h5py
    import numpy as np
    import pymultinest

    # variables
    gal      = sys.argv[1]
    zmin     = sys.argv[2]
    zmax     = sys.argv[3]
    pre      = sys.argv[4]
    GC       = 'NGC+SGC'
    if gal == 'LRG':
        rscale= 'log'
        rmin     = 5
        rmax     = 40
    else:
        rscale   = 'linear' # 'log'
        rmin     = 5
        rmax     = 65
    function =  'mps' # 'wp', 'mps', 'Pk'
    nseed    = 25
    date     = '0218'
    npoints  = 100 
    multipole= 'quad' # 'mono','quad','hexa'
    var      = 'Vpeak'  #'Vmax' 'Vpeak'
    Om       = 0.31
    boxsize  = 1000
    nthread  = 1
    autocorr = 1
    mu_max   = 1
    nmu      = 120
    autocorr = 1
    home     = '/home/astro/jiayu/Desktop/SHAM/'
    fileroot = '{}MCMCout/zbins_{}/{}{}_{}_{}_{}_z{}z{}/multinest_'.format(home,date,pre,function,rscale,gal,GC,zmin,zmax)
    cols = ['col4','col5']

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
    elif (rscale=='log'):
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

    # cosmological parameters
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+z)**3+Ode)

    # generate s bins
    bins  = np.arange(5,55,1)
    nbins = len(bins)-1
    s = (bins[:-1]+bins[1:])/2
    filename = 'param_evolution/SHAM_amplitude_z{}z{}.dat'.format(zmin,zmax)
    seqs = [i for i in range(nseed)]

    # SHAM halo catalogue
    print('reading the UNIT simulation snapshot with a(t)={}'.format(a_t))  
    halofile = home+'catalog/UNIT_hlist_'+a_t+'.hdf5'        
    read = time.time()
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
    half = int32(len(datac)/2)
    print(len(datac))
    print('read the halo catalogue costs {:.6}s'.format(time.time()-read))

    # UNIT 2pcf:
    mu = (np.linspace(0,mu_max,nmu+1)[:-1]+np.linspace(0,mu_max,nmu+1)[1:]).reshape(1,nmu)/2+np.zeros((nbins,nmu))
    RR_counts = 4*np.pi/3*(bins[1:]**3-bins[:-1]**3)/(boxsize**3)
    rr=((RR_counts.reshape(nbins,1)+np.zeros((1,nmu)))/nmu)

    # generate uniform random numbers
    print('generating uniform random number arrays...')
    uniform_randoms = [np.random.RandomState(seed=1000*x).rand(len(datac)).astype('float32') for x in range(nseed)] 

    # SHAM application
    def sham_cal(uniform,seq,sigma_high,sigma,v_high):
        # scatter Vpeak
        scatter = 1+append(sigma_high*sqrt(-2*log(uniform[:half]))*cos(2*pi*uniform[half:]),sigma_high*sqrt(-2*log(uniform[:half]))*sin(2*pi*uniform[half:]))
        scatter[scatter<1] = np.exp(scatter[scatter<1]-1)
        datav = datac[:,1]*scatter
        # select halos
        percentcut = int(len(datac)*v_high/100)
        LRGscat = datac[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
        datav = datav[argpartition(-datav,SHAMnum+percentcut)[:(SHAMnum+percentcut)]]
        LRGscat = LRGscat[argpartition(-datav,percentcut)[percentcut:]]

        np.savetxt('param_evolution/catalogues/SHAM-z{}z{}-seed{}.dat'.format(zmin,zmax,seq),LRGscat[:,-3:])
        
        # Corrfunc 2pcf and wp
        if function == 'mps':
            DD_counts = DDsmu(autocorr, nthread,bins,mu_max, nmu,LRGscat[:,2],LRGscat[:,3],LRGscat[:,4],periodic=True, verbose=False,boxsize=boxsize)
            mono = (DD_counts['npairs'].reshape(nbins,nmu)/(SHAMnum**2)/rr-1)
            xi0 = np.sum(mono,axis=-1)/nmu
        elif function == 'Pk':
            """
            # define 3D density field
            delta = np.zeros((grid,grid,grid), dtype=np.float32)
            # construct 3D density field
            MASL.MA(LRGscat[:,-3:].astype('float32'), delta, boxsize, MAS, verbose=True)
            # at this point, delta contains the effective number of particles in each voxel
            # now compute overdensity and density constrast
            delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0
            # compute power spectrum
            Pk = PKL.Pk(delta, boxsize, axis=0, MAS=MAS, threads=1, verbose=True)
            # 1D P(k)
            xi0     = Pk.Pk1D
            """
        f = open(filename,'a')
        np.savetxt(f,xi0[np.newaxis,:])
        f.close()

    # read the posterior file
    parameters = ["sigma","Vsmear","Vceil"]
    npar = len(parameters)
    a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)

    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_cal,list(zip(uniform_randoms,seqs,repeat(np.float32(a.get_best_fit()['parameters'][0])),repeat(np.float32(a.get_best_fit()['parameters'][1])),repeat(np.float32(a.get_best_fit()['parameters'][2]))))) 
        
else:
    #!/usr/bin/env python3
    import matplotlib 
    matplotlib.use('agg')
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    import pymultinest
    import sys

    from scipy.special import hyp2f1
    from scipy.misc import derivative

    function = 'Pk' #'mps' 'Pk'
    # bias evolution:
    #zmins = ['0.2', '0.33','0.43','0.51','0.57','0.6','0.6','0.65','0.7','0.8','0.2', '0.43','0.6']
    #zmaxs = ['0.33','0.43','0.51','0.57','0.7', '0.7','0.8','0.8' ,'0.9','1.0','0.43','0.7', '1.0']
    import matplotlib.gridspec as gridspec
    date = sys.argv[1] #'0218' '0729'
    plottype = sys.argv[2] #'vsmear' 'bias'

    fontsize=15
    plt.rc('font', family='serif', size=fontsize)
    if (date == '0218')&(plottype=='vsmear'):
        Ncol = 3
        fig = plt.figure(figsize=(10,5))
        spec = gridspec.GridSpec(nrows=1,ncols=2, left = 0.1,right = 0.98,bottom=0.15,top = 0.98, hspace=0.1,wspace=0.05)
        ax = np.empty((1,2), dtype=type(plt.axes))
    else:
        Ncol = 1
        fig = plt.figure(figsize=(5,4.5))
        spec = gridspec.GridSpec(nrows=1,ncols=1, left = 0.15,right = 0.98,bottom=0.12,top = 0.98, hspace=0.1,wspace=0.05)
        ax = np.empty((1,1), dtype=type(plt.axes))

    for Pind in range(Ncol):
        if (date == '0218'):
            parameters = ["sigma","Vsmear","Vceil"]
            if plottype == 'vsmear':
                if Pind == 1:
                    zmins = ['0.2', '0.33','0.43','0.51','0.57','0.6','0.65','0.7','0.8']#,'0.6'
                    zmaxs = ['0.33','0.43','0.51','0.57','0.7', '0.7','0.8' ,'0.9','1.0']#,'0.8'
                elif Pind ==0:
                    zmins = ['0.2', '0.43','0.6']
                    zmaxs = ['0.43','0.7', '1.0']    
                else:
                    parameters = ["sigma","Vsmear"]
                    zmins = ['0.2', '0.33','0.43','0.51', '0.2']
                    zmaxs = ['0.33','0.43','0.51','0.57','0.43']
            else:
                zmins = ['0.2', '0.33','0.43','0.51','0.57','0.6','0.65','0.7','0.8']#,'0.6'
                zmaxs = ['0.33','0.43','0.51','0.57','0.7', '0.7','0.8' ,'0.9','1.0']#,'0.8'                
        else:
            parameters = ["sigma","Vsmear"]
            zmins = ['0.2', '0.33','0.43','0.51']
            zmaxs = ['0.33','0.43','0.51','0.57']

    #zmins = ['0.6','0.6','0.65','0.7','0.8']
    #zmaxs = ['0.7','0.8','0.8' ,'0.9','1.0']
    #zmins = ['0.2', '0.2', '0.33','0.43','0.51']
    #zmaxs = ['0.33','0.43','0.43','0.51','0.57']

        SHAMbias   = []
        SHAMbiaserr= []
        zeff = []
        bounds0 = [];best0=[]
        bounds1 = [];best1=[]
        bounds2 = [];best2=[]
        bounds3 = [];best3=[]
        bounds4 = [];best4=[]

        home     = '/home/astro/jiayu/Desktop/SHAM/'
        
        Om = 0.31
        def D(a):
            return a * hyp2f1(1./3.,1,11./6.,a**3*(1-1./Om))
        def f(a):
            derv = derivative(D, a, dx=1e-3)
            return a * derv / D(a)

        for zmin,zmax in zip(zmins,zmaxs):
            if (zmin=='0.6')&(zmax=='1.0'): 
                SHAMnum   = int(6.26e4)
                gal = 'LRG'
                z = 0.7781
                a_t = '0.56220'
                ver = 'v7_2'
                rscale = 'linear'
                dvstd  = [111.6,115.1]
                dvnorm = [81.2,84.4]#[78.3,82.3]
            elif (zmin=='0.43')&(zmax=='0.51'): 
                gal = 'CMASS'
                SHAMnum = 342000
                z = 0.4686
                a_t = '0.68620'
                rscale = 'linear'
                dvstd  = [46.1,47.7]
                dvnorm = [38.6,39.4]#[38.6,39.4]
            elif zmin=='0.51':
                gal = 'CMASS'
                SHAMnum = 363000
                z = 0.5417 
                a_t = '0.64210'
                rscale = 'linear'
                dvstd  = [51.9,53.5]
                dvnorm = [43.0,43.8]#[43.0,43.8]
            elif zmin=='0.57':
                gal = 'CMASS'
                SHAMnum = 160000
                z = 0.6399
                a_t =  '0.61420'
                rscale = 'linear'
                dvstd  = [60.0,61.8]
                dvnorm = [46.5,47.7]#[46.5,47.7]
            elif (zmin=='0.43')&(zmax=='0.7'):  
                gal = 'CMASS'
                SHAMnum = 264000
                z = 0.5897
                a_t = '0.62800'
                rscale = 'linear'
                dvstd  = [53.7,54.7]
                dvnorm = [44.5,45.1]#[44.5,45.1]
            elif (zmin=='0.2')&(zmax=='0.33'):   
                gal = 'LOWZ'
                SHAMnum = 337000
                z = 0.2754
                a_t = '0.78370' 
                rscale = 'linear'
                dvstd  = [25.8,27.1]
                dvnorm = [20.9,21.5]#[20.9,21.5]
            elif zmin=='0.33':
                gal = 'LOWZ'
                SHAMnum = 258000
                z = 0.3865
                a_t = '0.71730'
                rscale = 'linear'
                dvstd  = [32.5,33.7]
                dvnorm = [27.5,28.2]#[27.5,28.2]
            elif (zmin=='0.2')&(zmax=='0.43'): 
                gal = 'LOWZ'
                SHAMnum = 295000
                z = 0.3441
                a_t = '0.74980'
                rscale = 'linear'
                dvstd  = [29.7,30.6]
                dvnorm = [23.8,24.3]#[23.8,24.3]
            elif (zmin=='0.6')&(zmax=='0.8'):
                gal='LRG'
                SHAMnum = int(8.86e4)
                z = 0.7051
                a_t = '0.58760'
                rscale = 'log'
                dvstd  = [103.2,106.4]
                dvnorm =  [80.5,85.3]#[81.0,85.1]
            elif (zmin=='0.6')&(zmax=='0.7'): 
                gal='LRG'           
                SHAMnum = int(9.39e4)
                z = 0.6518
                a_t = '0.60080'
                rscale = 'log'
                dvstd  =[100.7,103.9]
                dvnorm = [81.6,87.4]#[81.4,87.0]
            elif zmin=='0.65':
                gal='LRG'
                SHAMnum = int(8.80e4)
                z = 0.7273
                a_t = '0.57470'
                rscale = 'log'
                dvstd  = [103.8,107.5]
                dvnorm =  [82.4,87.6]#[82.5,87.3]
            elif zmin=='0.7':
                gal='LRG'
                SHAMnum = int(6.47e4)
                z=0.7968
                a_t = '0.54980'
                rscale = 'log'
                dvstd  = [112.4,117.1]
                dvnorm = [81.5,84.7]#[81.2,84.6]
            else:
                gal = 'LRG'
                SHAMnum = int(3.01e4)
                z= 0.8777
                a_t = '0.52600'
                rscale = 'log'
                dvstd  = [129.0,136.6]
                dvnorm = [83.1,89.1]#[85.0,89.9]
            zeff.append(z)
            if plottype == 'bias':
                if function == 'mps':
                    filename = 'param_evolution/real_space/SHAM_amplitude_z{}z{}.dat'.format(zmin,zmax)
                    columns = []
                    with open(filename, 'r') as td:
                        for line in td:
                            if line[0] == '#':
                                info = re.split(' +', line)
                                columns.append(info)
                    # best-fits: columns[0][4][:-1], columns[0][6][:-1], columns[0][8][:-1]
                    biasz = np.loadtxt(filename)
                    ind0 = 0
                    sbinmin = 5; sbinmax = 25
                    biasmean = np.sqrt(np.mean(biasz[sbinmin-(5-ind0):sbinmax-(5-ind0)],axis=1)/D(1/(1+z))**2)

                elif function =='Pk':
                    UNIT = np.loadtxt('param_evolution/UNIT_linear-interp.pk')
                    pkseed = []
                    for seed in range(25):
                        filename = 'param_evolution/catalogues_pk/z{}z{}-seed{}.pk'.format(zmin,zmax,seed)
                        pkseed.append(np.loadtxt(filename,usecols=5))
                    biasmean = np.sqrt(np.mean(pkseed/UNIT[:,-1]/D(float(a_t))**2,axis=1))

                SHAMbias.append(np.mean(biasmean))
                SHAMbiaserr.append(np.std(biasmean))
    
            
            # best-fits and their constraints
            if (date == '0218')&(plottype=='vsmear'):
                if Pind !=2:
                    fileroot = '{}MCMCout/zbins_0218/mps_{}_{}_NGC+SGC_z{}z{}/multinest_'.format(home,rscale,gal,zmin,zmax)
                else:
                    fileroot = '{}MCMCout/zbins_0729/mps_{}_{}_NGC+SGC_z{}z{}/multinest_'.format(home,rscale,gal,zmin,zmax)
            else:
                fileroot = '{}MCMCout/zbins_{}/mps_{}_{}_NGC+SGC_z{}z{}/multinest_'.format(home,date,rscale,gal,zmin,zmax)

            npar = len(parameters)
            a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
            stats = a.get_stats()
            ind = 0
            param = stats['marginals'][ind]['1sigma']
            best = a.get_best_fit()['parameters'][ind]
            bounds0.append(abs(param-best))
            best0.append(best)
            ind = 1
            param = stats['marginals'][ind]['1sigma']
            best = a.get_best_fit()['parameters'][ind]
            bounds1.append(abs(param-best))
            best1.append(best)
            if npar == 3:
                ind = 2
                param = stats['marginals'][ind]['1sigma']
                best = a.get_best_fit()['parameters'][ind]
                bounds2.append(abs(param-best))
                best2.append(best)
            # dv meaurements
            bounds3.append((dvstd[1]-dvstd[0])/2)
            best3.append(np.mean(dvstd))
            bounds4.append((dvnorm[1]-dvnorm[0])/2)
            best4.append(np.mean(dvnorm))

        # plot 
        if date == '0729':
            ax[0,Pind] = fig.add_subplot(spec[0,Pind])
            ax[0,Pind].errorbar(np.array(zeff),np.array(best0),np.array(bounds0).T,color='k', marker='o',ecolor='k',ls="none")
            ax[0,Pind].axvline(0.43, color= "k",linestyle='--')
            pos = 0.5
            ax[0,Pind].text(0.3, pos, 'LOWZ',size=fontsize)
            ax[0,Pind].text(0.48, pos, 'CMASS',size=fontsize)
            ax[0,Pind].set_ylabel('$\sigma$',size=fontsize)
            ax[0,Pind].set_xlabel('$z_{eff}$',size=fontsize)
            ax[0,Pind].set_ylim(0.2,0.55)
            plt.yticks([0.2,0.3,0.4,0.5])
            ax[0,Pind].set_xlim(0.23,0.6)
            savename = 'sigma_evolution_2param.pdf'
        elif plottype == 'bias':
            ax[0,Pind] = fig.add_subplot(spec[0,Pind])
            if function =='mps':
                srange = ': mps on 5-25Mpc/h'
            elif function == 'Pk':
                srange = ': P(k) linear range'
            #plt.title(r'SHAM bias evolution{}'.format(srange))
            ax[0,Pind].errorbar(np.array(zeff),np.array(SHAMbias),np.array(SHAMbiaserr),color='k', marker='o',ecolor='k',ls="none")
            ax[0,Pind].axvline(0.43, color= "k",linestyle='--')
            ax[0,Pind].axvline(0.645, color = "k",linestyle='--')
            pos = 3.2#0.625
            ax[0,Pind].text(0.25, pos, 'LOWZ',fontsize=fontsize)
            ax[0,Pind].text(0.48, pos, 'CMASS',fontsize=fontsize)
            ax[0,Pind].text(0.73, pos, 'eBOSS',fontsize=fontsize)
            ax[0,Pind].set_xlim(float(zmins[0]),float(zmaxs[-1])*0.95)
            ax[0,Pind].set_ylabel('linear bias',fontsize=fontsize)
            ax[0,Pind].set_xlabel('$z_{eff}$',fontsize=fontsize)
            savename = 'bias_evolution-{}.pdf'.format(function)
            #np.savetxt('param_evolution/SHAM-bias.txt',np.array([zeff,SHAMbias,SHAMbiaserr]).T,header='zeff bias biaserr')
        elif plottype == 'vsmear':
            # plot together
            if Pind !=2:
                ax[0,Pind] = fig.add_subplot(spec[0,Pind])

                ax[0,Pind].fill_between(np.array(zeff),np.array(best3)-np.array(bounds3),np.array(best3)+np.array(bounds3),color='r',alpha=0.4)
                if Pind ==0:
                    ax[0,Pind].errorbar(np.array(zeff),np.array(best1),np.array(bounds1).T,color='k', marker='o',ecolor='k',ls="none",label='3-param Vsmear')
                    ax[0,Pind].plot(np.array(zeff),np.array(best4),color='b', label='$\sigma_{\Delta v}$')
                    ax[0,Pind].plot(np.array(zeff),np.array(best3),color='r', label='std$_{\Delta v}$')  
                else:
                    ax[0,Pind].errorbar(np.array(zeff),np.array(best1),np.array(bounds1).T,color='k', marker='o',ecolor='k',ls="none",label='_hidden')
                    ax[0,Pind].plot(np.array(zeff),np.array(best4),color='b', label='_hidden')
                    ax[0,Pind].plot(np.array(zeff),np.array(best3),color='r', label='_hidden')  
                ax[0,Pind].fill_between(np.array(zeff),np.array(best4)-np.array(bounds4),np.array(best4)+np.array(bounds4),color='b',alpha=0.4)
                ax[0,Pind].axvline(0.43, color= "k",linestyle='--')
                ax[0,Pind].axvline(0.645, color = "k",linestyle='--')
                pos = 5
                ax[0,Pind].text(0.25, pos, 'LOWZ',fontsize=fontsize)
                ax[0,Pind].text(0.48, pos, 'CMASS',fontsize=fontsize)
                ax[0,Pind].text(0.74, pos, 'eBOSS',fontsize=fontsize)
                if Pind ==0:
                    ax[0,Pind].set_ylabel('$\Delta v$ (km/s)',fontsize=fontsize)#('$\\xi_0(gal)$/$\\xi_0(halo)$')
                else:
                    plt.yticks(alpha=0)
                ax[0,Pind].set_xlabel('$z_{eff}$',fontsize=fontsize)
                ax[0,Pind].set_ylim(0,pos*1.1)
                ax[0,Pind].set_xlim(float(zmins[0]),float(zmaxs[-1])*0.95)
            else:
                ax[0,1].errorbar(np.array(zeff[:-1]),np.array(best1[:-1]),np.array(bounds1[:-1]).T,color='k', marker='^',ecolor='k',ls="none",label='2-param Vsmear')
                #ax[0,0].legend(loc=2)
                ax[0,0].errorbar(zeff[-1],best1[-1],np.array(bounds1[-1]).reshape(2,1),color='k', marker='^',ecolor='k',ls="none",label='2-param Vsmear')
                ax[0,0].legend(loc=2)
            if Pind == 0:
                plt.yticks([0,40,80,120,160])
                plt.legend(loc=2,prop={"size":fontsize-1})
            plt.ylim(-10,180)
            savename = 'Vsmear_vs_dv_3param.pdf'

    plt.savefig(savename)
    plt.close()

"""
    from scipy.optimize import curve_fit
    def linear(x,a,b):
        return a*x+b
    popt, pcov = curve_fit(linear,1/(1+np.array(zeff)),np.array(best0),sigma=np.mean(abs(np.array(bounds0).T),axis=0))
    plt.title(r'The 2-param SHAM best-fit $\sigma$ slope {:.2f} $\pm$ {:.2f}'.format(popt[0],np.sqrt(np.diag(pcov))[0]))
    plt.errorbar(1/(1+np.array(zeff)),np.array(best0),np.array(bounds0).T,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(1/(1+np.array(zeff)),linear(1/(1+np.array(zeff)),*popt),'',label='best-fit')
    #plt.axvline(0.43, color= "k",linestyle='--')
    #plt.axvline(0.645, color = "k",linestyle='--')
    plt.axvline(0.7, color= "k",linestyle='--')
    plt.text(0.75, 0.1, 'LOWZ')
    plt.text(0.65, 0.1, 'CMASS')
    plt.ylabel('$\sigma$')#('$\\xi_0(gal)$/$\\xi_0(halo)$')
    #plt.xlabel('$z_{eff}$')
    plt.xlabel('$a(z_{eff}$)')
    plt.legend(loc=0)
    plt.ylim(0,0.6)
    plt.xlim(0.625,0.8)
    plt.savefig('sigma_evolution.png')
    plt.close()

    # parameter evolution fitting
    # linear function
    def linear(x,a,b):
        return a*x+b
    
    import matplotlib.gridspec as gridspec
    from scipy.optimize import curve_fit

    fig = plt.figure(figsize=(21,6))
    spec = gridspec.GridSpec(nrows=1,ncols=3,wspace=0.2)
    ax = np.empty((1,3), dtype=type(plt.axes))

    ax[0,0] = fig.add_subplot(spec[0,0])
    popt, pcov = curve_fit(linear,np.array(zeff),np.array(best0),sigma=np.mean(abs(np.array(bounds0).T),axis=0))
    plt.title(r'The SHAM best-fit $\sigma$ slope {:.2f} $\pm$ {:.2f}'.format(popt[0],np.sqrt(np.diag(pcov))[0]))
    plt.errorbar(np.array(zeff),np.array(best0),np.array(bounds0).T,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(np.array(zeff),linear(np.array(zeff),*popt),'',label='best-fit')
    plt.axvline(0.43, color= "k",linestyle='--')
    plt.axvline(0.645, color = "k",linestyle='--')
    pos = 1.0
    plt.text(0.3, pos, 'LOWZ')
    plt.text(0.5, pos, 'CMASS')
    plt.text(0.7, pos, 'eBOSS LRG')
    plt.ylabel('$\sigma$')#('$\\xi_0(gal)$/$\\xi_0(halo)$')
    plt.xlabel('$z_{eff}$')
    plt.legend(loc=0)
    plt.ylim(0,pos*1.2)

    ax[0,1] = fig.add_subplot(spec[0,1])
    popt, pcov = curve_fit(linear,np.array(zeff)[2:],np.array(best2)[2:],sigma=np.mean(abs(np.array(bounds2).T[:,2:]),axis=0))
    plt.title(r'The SHAM best-fit Vceil slope {:.2f} $\pm$ {:.2f}'.format(popt[0],np.sqrt(np.diag(pcov))[0]))
    plt.errorbar(np.array(zeff),np.array(best2),np.array(bounds2).T,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(np.array(zeff)[2:],linear(np.array(zeff)[2:],*popt),label='best-fit')
    plt.axvline(0.43, color= "k",linestyle='--')
    plt.axvline(0.645, color = "k",linestyle='--')
    pos = 0.1
    plt.text(0.3, pos, 'LOWZ')
    plt.text(0.5, pos, 'CMASS')
    plt.text(0.7, pos, 'eBOSS LRG')
    plt.ylabel('Vceil')#('$\\xi_0(gal)$/$\\xi_0(halo)$')
    plt.legend(loc=0)
    plt.ylim(0,pos*1.2)
    plt.xlabel('$z_{eff}$')

    ax[0,2] = fig.add_subplot(spec[0,2])
    popt, pcov = curve_fit(linear,np.array(zeff)[2:],np.array(best1)[2:],sigma=np.mean(abs(np.array(bounds1).T[:,2:]),axis=0))
    plt.title(r'The SHAM best-fit Vsmear slope {:.2f} $\pm$ {:.2f}'.format(popt[0],np.sqrt(np.diag(pcov))[0]))
    plt.errorbar(np.array(zeff),np.array(best1),np.array(bounds1).T,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(np.array(zeff)[2:],linear(np.array(zeff)[2:],*popt),label='best-fit')
    plt.axvline(0.43, color= "k",linestyle='--')
    plt.axvline(0.645, color = "k",linestyle='--')
    pos = 160
    plt.text(0.3, pos, 'LOWZ')
    plt.text(0.5, pos, 'CMASS')
    plt.text(0.7, pos, 'eBOSS LRG')
    plt.ylim(0,pos*1.2)
    plt.ylabel('Vsmear')#('$\\xi_0(gal)$/$\\xi_0(halo)$')
    plt.xlabel('$z_{eff}$')
    plt.legend(loc=0)
    plt.savefig('parameter_evolution.png')
    plt.close()

"""
"""
# save the multinest chain, which doesn't mean anything
# calculate the SHAM 2PCF
def sham_2pcf(index,filename):
    print('sigma, Vsmear, Vceil, -0.5*chi2: ',A[index,:])

    with Pool(processes = nseed) as p:
        xi1_ELG = p.starmap(sham_cal,list(zip(uniform_randoms,repeat(A[index,0]),repeat(A[index,1]),repeat(A[index,2])))) 
    
    # xi0
    mean0 = np.mean(xi1_ELG,axis=0)

    f = open(filename,'a')
    bias = np.zeros(0)
    bias = np.append(A[index,3],mean0)
    bias = np.append(A[index,2],bias)
    bias = np.append(A[index,1],bias)
    bias = np.append(A[index,0],bias)

    np.savetxt(f,np.array(bias)[np.newaxis,:])
    f.close()

import pymultinest
# read the posterior file
parameters = ["sigma","Vsmear","Vceil"]
npar = len(parameters)
a = pymultinest.Analyzer(npar, outputfiles_basename = fileroot)
A = a.get_equal_weighted_posterior()

nbias = 200
chunk = len(A)//nbias
#import pdb;pdb.set_trace()
def biasbar(biasarr,sbinmin,sbinmax):
    return np.array([biasarr[:,0],biasarr[:,2],np.mean(biasarr[:,sbinmin-(5-ind0):sbinmax-(5-ind0)],axis=1)]).T

if function == 'mps':
    sham_2pcf(np.where(A[:,-1]==a.get_best_fit()['log_likelihood'])[0][0],'param_evolution/chain-amplitude_{}_z{}z{}.dat'.format(gal,zmin,zmax))

    for ind in range(nbias):
        sham_2pcf(-1-chunk*ind,'param_evolution/chain-amplitude_{}_z{}z{}.dat'.format(gal,zmin,zmax))
        if (ind+1)%(nbias/10)==0:
            print('{}% completed'.format(ind+1))

    # plot chains
    Uxi00 = np.loadtxt('param_evolution/UNIT_'+a_t+'.mps',usecols=3)
    print('the UNIT 5-65Mpc/h monopoles are ready.')
    bias = np.loadtxt('param_evolution/chain-amplitude_{}_z{}z{}.dat'.format(gal,zmin,zmax))
    ind0 = 4
    bias[:,ind0:]/=Uxi00
    # choose linear range
    fig,ax = plt.subplots()
    plt.title('bias_z{}z{}'.format(zmin,zmax))
    for i in range(5):
        ax.plot(s,bias[i,ind0:])
    plt.savefig('param_evolution/bias_z{}z{}.png'.format(zmin,zmax))
    plt.close()
    # contour: sigma-Vceil vs bias
    labels = ["sigma","Vceil"]#,"bias"]
    import pdb;pdb.set_trace()
    samples1 = MCSamples(samples=biasbar(bias,5,25)[:,:-1],loglikes = biasbar(bias,5,25)[:,-1],names = labels,label='5-25 bias')
    samples2 = MCSamples(samples=biasbar(bias,25,55)[:,:-1],loglikes = biasbar(bias,25,55)[:,-1],names = labels,label='25-55 bias')

    g = plots.getSinglePlotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g = plots.getSubplotPlotter()
    g.triangle_plot([samples1,samples2],labels, filled=True)
    g.export('param_evolution/bias-contour_{}_z{}z{}.png'.format(gal,zmin,zmax))
    plt.close()
elif function == 'Pk':
    sham_2pcf(np.where(A[:,-1]==a.get_best_fit()['log_likelihood'])[0][0],'param_evolution/Pk_z{}z{}.dat'.format(zmin,zmax))
    Uxi00 = np.loadtxt('param_evolution/Pklin_z{}z{}.dat'.format(zmin,zmax))
    print('the UNIT P(k) are ready.')
    bias = np.loadtxt('param_evolution/Pk_z{}z{}.dat'.format(zmin,zmax))
    ind0 = 4
    bias[ind0:]/=Uxi00
    # choose linear range
    fig,ax = plt.subplots()
    plt.title('bias_z{}z{}'.format(zmin,zmax))
    k = np.linspace(2*np.pi/boxsize,np.pi*grid/boxsize,int(grid/2))
    plt.loglog(k,bias[ind0:])
    plt.savefig('bias_z{}z{}.png'.format(zmin,zmax))
    plt.close()
"""


"""
import MAS_library as MASL
import Pk_library as PKL

from nbodykit.lab import cosmology
cosmo = cosmology.Cosmology()
new_cosmo = cosmo.match(Omega0_m=0.31)
c = cosmology.Planck15

# density field parameters
grid    = 512    #the 3D field will have grid x grid x grid voxels
MAS     = 'CIC'  #mass-assigment scheme

# trying to calculate P(k), but failed
# LRGscat = SHAM catalogue, columns: Vpeak(km/s), Vz(km/s), X, Y, Z (Mpc/h)
grid = 512
boxsize = 1000
MAS     = 'CIC'  #mass-assigment scheme

delta = np.zeros((grid,grid,grid), dtype=np.float32)
MASL.MA(LRGscat[:,-3:].astype('float32'), delta, boxsize, MAS, verbose=True)
delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

mesh = ArrayMesh(delta,boxsize)
r = FFTPower(mesh,mode='1d',dk=0.005,kmin=0.01)
#Pk = PKL.Pk(delta, boxsize, axis=0, MAS=MAS, threads=1, verbose=True)
Pk     = r.power
xi0 = Pk['power'].real-Pk.attrs['shotnoise']
k = Pk['k']
plt.loglog(k,xi0)
plt.savefig('halo_nbody.png')
"""