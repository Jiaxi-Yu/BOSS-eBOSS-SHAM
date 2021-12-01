from astropy.io import fits
from astropy.table import Table
from glob import glob
from multiprocessing import Pool
import pylab as plt
import numpy as np
import matplotlib.gridspec as gridspec
import sys
import h5py

task = sys.argv[1]
# 'format-convert', 'rsd', 'covariance','cosmosim','mock0smear', bin5Mpch'

# zeff and file directory
z    = 0.74
home = '/global/homes/j/jiaxi/DESIVsmear/catalogues/'
#'/global/cscratch1/sd/jiaxi/desi/LSS/Sandbox/Vsmear/catalogues/'
sourcefile = '/global/project/projectdirs/desi/cosmosim/UNIT-BAO-RSD-challenge/Stage2Recon/UNITSIM/LRG/LRG-wpmax-v3-snap103-redshift{}_dens0.dat'.format(z)
datadir = '/global/cscratch1/sd/jiaxi/SHAM/'
# load data and set cosmology
boxsize=1000
Om = 0.31
Ode = 1-Om
H = 100*np.sqrt(Om*(1+z)**3+Ode)

if task == 'format-convert':
    # convert hdf5 -> ascii, not in interactive nodes
    #mocksfiles = glob()
    mocksfiles = '/global/cfs/cdirs/desi/mocks/GLAM/GLAM_LRG_TSRongpu/PMILL/z0.74/DESI_GLAM_LRG_z0.74_HOD_{}.hdf5'
    nfile  = 100 #len(mocksfiles)

    def datatrans(mockdir):
        hf=h5py.File(mockdir,"r")
        pos = hf.get('pos')[:]
        vel = hf.get('vel')[:]
        
        cata = []
        cata = np.hstack((pos,vel[:,-1][:,np.newaxis]))
        np.savetxt('{}GLAM_0.74/mocks/{}.dat'.format(datadir,mockdir[88:-5]),cata)
        hf.close()
        #return cata
        
    for n in range(nfile):
        datatrans(mocksfiles.format(n))
        #print(n)
        if n==0:
            print('mock reading&2PCF calculation start')
        elif (n+1)%(np.ceil(nfile/10))==0:
            print('z = {} has finished {}%'.format(z,(n+1)//(nfile/100)))    

elif task == 'rsd':
    # real -> redshift
    mocksdat = glob(datadir+'GLAM_0.74/mocks/*.dat')
    nfile  = len(mocksdat)

    def rsd(mockdir):
        mockdata = np.loadtxt(mockdir)
        tmp = mockdata[:,2] + mockdata[:,-1]*(1+z)/H
        mockdata[:,2] = tmp%boxsize
        print(datadir+'GLAM_0.74/mocks_rsd/{}'.format(mockdir[48:]))
        np.savetxt(datadir+'GLAM_0.74/mocks_rsd/{}'.format(mockdir[48:]),mockdata[:,(0,1,2)])
        
        
    pool = Pool()     
    for n, temp_array in enumerate(pool.imap(rsd,mocksdat)):
        #print(n)
        finished = temp_array
        if n==0:
            print('mock reading&2PCF calculation start')
        elif (n+1)%(np.ceil(nfile/10))==0:
            print('z = {} has finished {}%'.format(z,(n+1)//(nfile/100)))
    pool.close() 
    pool.join()

elif task == 'covariance':
    # extract mock mps
    from functools import partial

    def mockextract(mockdir,Pk=False):
        if Pk:
            mono,quad,hexa  = np.loadtxt(mockdir,usecols=(5,6,7),unpack=True)
        else:
            mono,quad,hexa  = np.loadtxt(mockdir,usecols=(3,4,5),unpack=True)
        return [mono,quad,hexa]
    
    for datatype in ['mps','Pk']:
        if datatype=='mps':
            tail = 'xi'
            mpsextract= partial(mockextract,Pk=False)
        else:
            tail = 'pk'
            mpsextract= partial(mockextract,Pk=True)

        mocksALL = glob(home+'/GLAM_0.74/{}/*.{}'.format(datatype,tail))
        nfile  = len(mocksALL)
        mpsmocks = [n for n in range(nfile)]

        pool = Pool()     
        for n, temp_array in enumerate(pool.imap(mpsextract,mocksALL)):
            mpsmocks[n] = temp_array
            if n==0:
                print('mock reading&2PCF calculation start')
            elif (n+1)%(np.ceil(nfile/10))==0:
                print('z = {} has finished {}%'.format(z,(n+1)//(nfile/100)))
        pool.close() 
        pool.join()
        #import pdb;pdb.set_trace()
        mono = np.array([mpsmocks[i][0] for i in range(nfile)]).T
        np.savetxt(home+'GLAM_0.74/mono_mocks.{}'.format(tail),mono)
        quadru = np.array([mpsmocks[i][1] for i in range(nfile)]).T
        np.savetxt(home+'GLAM_0.74/quad_mocks.{}'.format(tail),quadru)
        hexadeca = np.array([mpsmocks[i][2] for i in range(nfile)]).T
        np.savetxt(home+'GLAM_0.74/hexa_mocks.{}'.format(tail),hexadeca)

elif task == 'cosmosim':
    # bash script: calculate and plot the multipoles with different Vsmear
    catafile = home+'cosmosim/cosmosim'

    names = ['nosmear','lorentzian','gaussian','stdev']
    cols = ["[\$1,\$2,\$4]","[\$1,\$2,\$5]","[\$1,\$2,\$6]","[\$1,\$2,\$7]"]
    for name,col in zip(names,cols):
        f = open('cosmosim-z0.74_jobs.sh','a')
        f.write('srun -n 1 -c 64 /global/homes/j/jiaxi/codes/FCFC/FCFC_2PT_BOX --conf {}fcfc_2pt_box.conf --input "{}" -P "{}" -E "{}" -M "{}" -x "{}"; '.format(home,catafile+'-z0.74.dat',catafile+'_{}.dd'.format(name),catafile+'_{}.xi'.format(name),catafile+'_{}.mps'.format(name),col))
        f.close()

elif task == 'mock0smear':
    catafile = home+'GLAM_0.74/mock0_smear/mock0'
    # bash script: calculate and plot the multipoles with different Vsmear
    names = ['nosmear','lorentzian','gaussian','stdev','lorentzian_trunc']
    cols = ["[\$1,\$2,\$3]","[\$1,\$2,\$4]","[\$1,\$2,\$5]","[\$1,\$2,\$6]","[\$1,\$2,\$7]"]
    for name,col in zip(names,cols):
        f = open('catalogues/GLAM_0.74/mock0_jobs.sh','a')
        f.write('srun -n 1 -c 64 /global/homes/j/jiaxi/codes/FCFC/FCFC_2PT_BOX --conf {}fcfc_2pt_box.conf --input "{}" -P "{}" -E "{}" -M "{}" -x "{}"; '.format(home,catafile[:-6]+'.dat',catafile+'_{}.dd'.format(name),catafile+'_{}.xi'.format(name),catafile+'_{}.mps'.format(name),col))
        f.close()

elif task == 'bin5Mpch':
    pass

elif task == 'plot':
    for datatype in ['mps','Pk']:
        if datatype=='mps':
            tail = 'xi'
            binmin = 1;binmax =30
            errmono   = np.loadtxt(home+'GLAM_0.74/mono_mocks.{}'.format(tail))[binmin:binmax]
            errquad   = np.loadtxt(home+'GLAM_0.74/quad_mocks.{}'.format(tail))[binmin:binmax]
            errhexa   = np.loadtxt(home+'GLAM_0.74/hexa_mocks.{}'.format(tail))[binmin:binmax]        
        else:
            tail = 'pk'
            errmono   = np.loadtxt(home+'GLAM_0.74/mono_mocks.{}'.format(tail))
            errquad   = np.loadtxt(home+'GLAM_0.74/quad_mocks.{}'.format(tail))
            errhexa   = np.loadtxt(home+'GLAM_0.74/hexa_mocks.{}'.format(tail))
        # errors from GLAM
        errbar  = np.std(np.vstack((errmono,errquad,errhexa)),axis=1)

        ## central mps/Pk calculation: average
        catafile = home+'GLAM_0.74/mock0_smear/'
        names = ['nosmear','lorentzian','gaussian','stdev','lorentzian_trunc2','data_like']
        monodata = [[],[],[],[],[],[]]
        quaddata = [[],[],[],[],[],[]]
        hexadata = [[],[],[],[],[],[]]
        for j,name in enumerate(names):
            for index in range(40):
                if datatype=='mps':
                    mpsdata  = np.loadtxt(catafile+'{}/mps_{}_{}.{}'.format(datatype,name,index,tail),usecols=(0,3,4,5))[binmin:binmax]
                else:            
                    mpsdata  = np.loadtxt(catafile+'{}/pk_{}_{}.{}'.format(datatype,name,index,tail),usecols=(0,5,6,7))
                monodata[j].append(mpsdata[:,1])
                quaddata[j].append(mpsdata[:,2])
                hexadata[j].append(mpsdata[:,3])
        s = mpsdata[:,0]
        if datatype=='mps':
            norm = s**2
            xlabel  = 's (Mpc $h^{-1}$)'
        else:
            norm = s**1.5           
            xlabel  = 'k ($h$ Mpc$^{-1}$)'
        nbins = len(s)
        mps = [np.mean(monodata,axis=1),np.mean(quaddata,axis=1),np.mean(hexadata,axis=1)]
        #import pdb;pdb.set_trace()

        # plot the 2PCF multipoles  
        plt.rc('font', family='serif', size=18) 
        fontsize=18
        fig = plt.figure(figsize=(20,8))
        spec = gridspec.GridSpec(nrows=2,ncols=3, height_ratios=[2, 1], wspace=0.23,hspace=0,left=0.05, right=0.98)
        ax = np.empty((2,3), dtype=type(plt.axes))
        for k,name in enumerate(['monopole','quadrupole','hexadecapole']):
            values=[np.zeros(nbins), mps[k][0]]       
            err   = [np.ones(nbins),norm*errbar[k*nbins:(k+1)*nbins]]
            for j in range(2):
                ax[j,k] = fig.add_subplot(spec[j,k])
                #ax[j,k].errorbar(s,norm*(mps[k][0]-values[j])/err[j],norm*errbar[k*nbins:(k+1)*nbins]/err[j],color='k', marker='o',ecolor='k',ls="none",label='nosmear 1$\sigma$')
                ax[j,k].plot(s,norm*(mps[k][0]-values[j])/err[j],color='k',label='no smear')
                ax[j,k].fill_between(s,norm*(mps[k][0]-values[j])/err[j]-norm*errbar[k*nbins:(k+1)*nbins]/err[j],norm*(mps[k][0]-values[j])/err[j]+norm*errbar[k*nbins:(k+1)*nbins]/err[j],color = 'k',alpha=0.2,label='_hidden')
                #ax[j,k].plot(s,norm*(mps[k][1]-values[j])/err[j],alpha=0.8,label='lorentzian')
                ax[j,k].plot(s,norm*(mps[k][2]-values[j])/err[j],alpha=0.8,label='gaussian')
                ax[j,k].plot(s,norm*(mps[k][3]-values[j])/err[j],alpha=0.8,label='stdev')
                ax[j,k].plot(s,norm*(mps[k][4]-values[j])/err[j],alpha=0.8,label='lorentzian < 200km/s')
                ax[j,k].plot(s,norm*(mps[k][5]-values[j])/err[j],alpha=0.8,label='data')
                
                if (j==0):
                    if datatype=='mps':
                        ax[j,k].set_ylabel('$s^2 * \\xi_{}$'.format(k*2),fontsize=fontsize)
                    else:
                        ax[j,k].set_ylabel('k$^{{1.5}}$ * $P_{}$'.format(k*2),fontsize=fontsize)

                    if k==0:
                        plt.legend(loc=0)
                    #plt.title('correlation function {}'.format(name))
                if (j==1):
                    if datatype=='mps':
                        ax[j,k].set_ylabel('$\Delta\\xi_{}$/err'.format(k*2),fontsize=fontsize)
                    else:
                        ax[j,k].set_ylabel('$\Delta P_{}$/err'.format(k*2),fontsize=fontsize)
                    plt.xlabel(xlabel)


                    #ax[j,k].plot(s,s**2*(xi[:,k]-values[j])/err[j],c='c',alpha=0.8)

                    #if k==0:
                    #    plt.ylim(-2,2)
                    #elif k==1:
                    #    plt.ylim(-2,6)

        plt.savefig(home+'mock0_{}.png'.format(datatype))
        plt.close()


"""
# add Vsmear for Jamie's catalogue
for j,redshift in enumerate([0.4573,0.7018]):
    if j==1:
        gamma =  30.237529380473834
        sigma =  33.35044873601363
        stdev =  57.501027544206146
    else:
        gamma =  18.778114022313883
        sigma =  21.224486490657547
        stdev =  36.83156583248373
    
    boxsize=1000
    Om = 0.307
    Ode = 1-Om
    H = 100*np.sqrt(Om*(1+redshift)**3+Ode)
    for gc in ['N','S']:
        sourcefile = '/global/u1/j/jdonaldm/SV3/LRG_v0.1_mocks/SV3_LRG_{}_angup_z-{}-MDPL2-mock.txt'.format(gc,redshift)

        # Vsmear random numbers
        data = np.loadtxt(sourcefile)
        random_lorentzian = cauchy.rvs(loc=0, scale=gamma, size=len(data))
        random_gaussian      = np.random.normal(loc=0,scale = sigma,size = len(data))
        random_stdev      = np.random.normal(loc=0,scale = stdev,size = len(data))
        vsmear = []

        print('generating Vsmeared peculier velocities')
        for k,randoms in enumerate([random_lorentzian,random_gaussian,random_stdev]):
            vsmear.append((data[:,-2]+(randoms*(1+redshift)/H)%boxsize)%boxsize)
        np.savetxt('/global/cscratch1/sd/jiaxi/Jamie/SV3_LRG_{}_angup_z-{}-MDPL2-mock.txt'.format(gc,redshift),np.hstack((data,np.array(vsmear).T)))

"""