import time
init = time.time()
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        help="data preparation: obs,zeff,neff",
        type=str,
        default=None,
        required=True,
    )
    args = None
    args = parser.parse_args()
    return args
args = parse_args()

home  = '/global/cscratch1/sd/jiaxi/SHAM/catalog/DESItest/'
if args.task == 'obs-2021':
    # create observational files for tests
    for datatype,tail,pair in zip(['XI02','WP'],['dat','wp'],['xi0-rmu','wp-rp-pi']):
        targets = [home+'data-202112/']#,home+'data-202110/']
        mpsdates= ['XI_17DEC']#,'XI_11Oct']
        #if tail == 'wp':
        #    targets,mpsdates = [home+'data-202112/'],['XI_17DEC']
        for target,mpsdate in zip(targets,mpsdates):
            #gals = ['LRG','ELG','BGS_BRIGHT','QSO']
            gals = ['LRG_main','ELG_HIP']+['cross-LRGxELG','cross-LRGxQSO','cross-ELGxQSO']
            #if mpsdate == 'XI_11Oct':
            #    gals = ['LRG','ELG']
            for gal in gals:
                origin = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/LSScats/test_Shadab/{}/{}/'.format(mpsdate,datatype)#XI_17DEC  XI_11Oct
                if   (gal[:3] == 'LRG')|(gal =='cross-LRGxELG'):
                    zmins = [0.4,0.6,0.8,0.6]
                    zmaxs = [0.6,0.8,1.1,1.1]
                elif (gal[:3] == 'ELG')|(gal == 'cross-ELGxQSO'):
                    zmins = [0.8,1.1,0.8]
                    zmaxs = [1.1,1.5,1.5]
                elif gal == 'cross-LRGxQSO':
                    zmins = [0.6,0.8,0.6]
                    zmaxs = [0.8,1.1,1.1]
                elif gal == 'BGS_BRIGHT':
                    zmins = [0.1,0.2,0.3,0.1]
                    zmaxs = [0.3,0.4,0.5,0.5]
                elif gal == 'QSO':
                    zmins = [0.8,1.1,1.5,0.8]
                    zmaxs = [1.1,1.5,2.1,2.1]
                    if tail == 'wp':
                        pair = 'wp-logrp-pi'
                for zmin,zmax in zip(zmins,zmaxs):
                    data = []
                    obsdir = '{}{}_NS_CLUSTERING_wEdWsys_z1z2_{}-{}_pip_angup-{}-NJN-240.txt'.format(origin,gal,zmin,zmax,pair)
                    if not os.path.exists('{}{}_z{}z{}.{}'.format(target,gal,zmin,zmax,tail)):
                        #print(os.path.exists(obsdir))
                        # append the sbins, xi0/wp for output
                        obsraw0 = np.loadtxt(obsdir)[:25]
                        data.append(obsraw0[:,0])
                        data.append(obsraw0[:,3])
                        # save truncated xi0/wp for all jackknife in "tot" and error in "err"
                        tot1 = obsraw0[5:,4:]
                        err1 = obsraw0[5:,2]
                        if tail == 'dat':
                            # read & save complete xi0 & xi2 and truncated covR
                            obsdir = '{}{}_NS_CLUSTERING_wEdWsys_z1z2_{}-{}_pip_angup-xi2-rmu-NJN-240.txt'.format(origin,gal,zmin,zmax)
                            obsraw2 = np.loadtxt(obsdir)[:25]
                            # append the xi2, xi0err, xi2err for output
                            data.append(obsraw2[:,3])
                            data.append(obsraw0[:,2])
                            data.append(obsraw2[:,2])
                            # save truncated xi2 for all jackknife in "tot" and error in "err"
                            tot2 = obsraw2[5:,4:]
                            err2 = obsraw2[5:,2]

                            # calculate covariance matrix
                            mocks = np.vstack((tot1,tot2))
                            print(np.std(mocks,axis=1)*np.sqrt(239)-np.append(err1,err2))
                            Nbins = mocks.shape[0]
                            Nmock = mocks.shape[1]
                            covcut  = np.cov(mocks)*239
                            covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
                            np.savetxt('{}{}-covR_z{}z{}.{}'.format(target,gal,zmin,zmax,tail),covR)
                        else:
                            # save wp error in "err"
                            err1 = obsraw0[:,2]
                            data.append(err1)

                        # save observation
                        np.savetxt('{}{}_z{}z{}.{}'.format(target,gal,zmin,zmax,tail),np.array(data).T)
elif args.task == 'obs-fuji':
    from pycorr import TwoPointCorrelationFunction
    # Fuji version 3
    target = 'data-fuji_3'
    for directory,tail in zip(['smu/xipoles_','rppi/wp_'],['dat','wp']):
        origin = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3/xi/{}'.format(directory)
        gals   = ['LRG','ELG','BGS_ANY','QSO'] \
                +['LRG_main','ELG_HIP','BGS_BRIGHT']#,'ELG_HIPnotqso']
        """    
        gals   = ['ELG_LRG','QSO_LRG'] \
                +['ELG_HIP_LRG','ELG_LRG_main','QSO_LRG_main']\#,'ELG_HIPnotqso_LRG'] \
                +['ELG_HIP_LRG_main']#,'ELG_HIPnotqso_LRG_main']\
                #+['ELG_HIPnotqso_QSO']
        for gal in gals:
            if gal != 'ELG_HIPnotqso_QSO':
                zmins = [0.8]
                zmaxs = [1.1]
            else:
                zmins = [0.8,1.1,0.8]
                zmaxs = [1.1,1.5,1.5]
        """
        for gal in gals:
            if   (gal[:3] == 'LRG'):
                zmins = [0.4,0.6,0.8,0.4]
                zmaxs = [0.6,0.8,1.1,1.1]
            elif (gal[:3] == 'ELG'):
                zmins = [0.8,1.1,0.8]
                zmaxs = [1.1,1.5,1.5]
            elif gal == 'BGS_BRIGHT':
                zmins = [0.1,0.2,0.3,0.1]
                zmaxs = [0.3,0.4,0.5,0.5]
            elif gal == 'QSO':
                zmins = [0.8,1.1,1.5,0.8]
                zmaxs = [1.1,1.5,2.1,2.1]
                #if tail == 'wp':
                #    pair = 'wp-logrp-pi'

            for zmin,zmax in zip(zmins,zmaxs):
                data = []
                obsdir = origin+'{}_{}_default_angular_bitwise_lin1_njack120.txt'.format(zmin,zmax)
                if not os.path.exists(home+'{}/{}_z{}z{}.{}'.format(target,gal,zmin,zmax,tail)):
                    if tail == 'dat':
                        # read & save complete xi0 & xi2 and truncated covR
                        data = np.loadtxt(obsdir,usecols=(0,2,3,5,6))[:25]
                        # save truncated xi2 for all jackknife in "tot" and error in "err"
                        tot1 = data[5:,1]
                        tot2 = data[5:,2]
                        err1 = data[5:,3]
                        err2 = data[5:,4]

                        # calculate covariance matrix
                        result = TwoPointCorrelationFunction.load(fn)

                        mocks = np.vstack((tot1,tot2))
                        print(np.std(mocks,axis=1)*np.sqrt(239)-np.append(err1,err2))
                        Nbins = mocks.shape[0]
                        Nmock = mocks.shape[1]
                        covcut  = np.cov(mocks)*239
                        covR  = np.linalg.pinv(covcut)*(Nmock-Nbins-2)/(Nmock-1)
                        np.savetxt('{}{}-covR_z{}z{}.{}'.format(target,gal,zmin,zmax,tail),covR)
                    else:
                        data = np.loadtxt(obsdir,usecols=(0,2,3,5,6))[5:25]

                    # save observation
                    np.savetxt(home+'{}/{}_z{}z{}.{}'.format(target,gal,zmin,zmax,tail),data)

elif args.task == 'zeff-simple-fuji':
    ZEFF = Table.read('DESI_SV3_neff.dat',format='ascii')
    root = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3/'
    gals   = ['LRG','ELG','BGS_ANY','QSO'] \
            +['LRG_main','ELG_HIP','BGS_BRIGHT']#,'ELG_HIPnotqso']

    zw = {}
    zeff = []
    # galaxy types, redshift ranges, and P0 values
    for gal in gals:
        if   (gal[:3] == 'LRG'):
            zmins = [0.4,0.6,0.8,0.6]
            zmaxs = [0.6,0.8,1.1,1.1]
            P0=1e4
        elif (gal[:3] == 'ELG'):
            zmins = [0.8,1.1,0.8]
            zmaxs = [1.1,1.5,1.5]
            P0=4e3
        elif (gal[:3] == 'BGS'):
            zmins = [0.1,0.2,0.3,0.1]
            zmaxs = [0.3,0.4,0.5,0.5]
            P0=7e3
        elif gal== 'QSO':
            zmins = [0.8,1.1,1.5,0.8]
            zmaxs = [1.1,1.5,2.1,2.1]
            P0=6e3   
        # read redshift and weights
        for zmin,zmax in zip(zmins,zmaxs):
            for GC in ['','N_','S_']:
                filename = root+'{}_{}clustering.dat.fits'.format(gal,GC)
                hdu      = fits.open(filename)
                data     = hdu[1].data
                hdu.close()
                # save z and w
                zw[gal+'_'+GC+'z'] = np.copy(data['Z'])
                zw[gal+'_'+GC+'w'] = np.copy(data['WEIGHT']/(1+data['NZ']*P0))
                # zeff calculation
                sel = (zw[gal+'_'+GC+'z']>zmin)&(zw[gal+'_'+GC+'z']<=zmax)
                zeff.append(sum(zw[gal+'_'+GC+'z'][sel]*zw[gal+'_'+GC+'z'][sel]**2)/sum(zw[gal+'_'+GC+'z'][sel]**2))
    ZEFF['zeff-simple'] = np.array(zeff)
    ZEFF.write('DESI_SV3_neff_zeff.dat',format='ascii')
elif args.task == 'zeff-pair-fuji':
    ZEFF   = Table.read('DESI_SV3_neff_zeff.dat',format='ascii')
    root   = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3/'
    output = '/global/cscratch1/sd/jiaxi/SHAM/catalog/DESItest/data-fuji_3-catalogues/'
    gals   = ['LRG','ELG','BGS_ANY','QSO'] \
            +['LRG_main','ELG_HIP','BGS_BRIGHT']#,'ELG_HIPnotqso']

    if not os.path.exists(output+'BGS_BRIGHT_clustering.dat'):
        # galaxy types, redshift ranges, and P0 values
        for gal in gals:
            if   (gal[:3] == 'LRG'):
                P0=1e4
            elif (gal[:3] == 'ELG'):
                P0=4e3
            elif (gal[:3] == 'BGS'):
                P0=7e3
            elif gal== 'QSO':
                P0=6e3  
            # read redshift and weights
            for GC in ['','N_','S_']:
                filename = root+'{}_{}clustering.dat.fits'.format(gal,GC)
                hdu      = fits.open(filename)
                data     = hdu[1].data
                hdu.close()    
                # save [X,Y,Z,Wtot]
                DATA = [data['RA'],data['DEC'],data['Z'],data['WEIGHT']/(1+data['NZ']*P0)]
                np.savetxt(output+'{}_{}clustering.dat'.format(gal,GC),np.array(DATA).T)
    else:
        FCFCdir = '/global/u2/z/zhaoc/programs/FCFC'
        zeff = []
        for gal in gals:
            if   (gal[:3] == 'LRG'):
                zmins = [0.4,0.6,0.8,0.6]
                zmaxs = [0.6,0.8,1.1,1.1]
                P0=1e4
            elif (gal[:3] == 'ELG'):
                zmins = [0.8,1.1,0.8]
                zmaxs = [1.1,1.5,1.5]
                P0=4e3
            elif (gal[:3] == 'BGS'):
                zmins = [0.1,0.2,0.3,0.1]
                zmaxs = [0.3,0.4,0.5,0.5]
                P0=7e3
            elif gal == 'QSO':
                zmins = [0.8,1.1,1.5,0.8]
                zmaxs = [1.1,1.5,2.1,2.1]
                P0=6e3
            # calculate the neff        
            for zmin,zmax in zip(zmins,zmaxs):            
                for GC in ['','N_','S_']:
                    infile = output+'{}_{}clustering.dat'.format(gal,GC)
                    outfile = output+'zeff-pair/{}_{}clustering.dat'.format(gal,GC)
                    if not os.path.exists(output+'zeff-pair/{}_z{}z{}.dd'.format('BGS_BRIGHT_S',0.1,0.5)):
                        f = open('zeff-pair.sh','a')
                        f.write('{}/FCFC_2PT      --conf {}/fcfc_2pt.conf -i {} -s \"\$3>={}&&\$3<{}\" -P {}_z{}z{}.dd \n'.format(FCFCdir,output,infile,zmin,zmax,outfile[:-15],zmin,zmax))
                        f.write('{}_zeff/FCFC_2PT --conf {}/fcfc_2pt.conf -i {} -s \"\$3>={}&&\$3<{}\" -P {}_z{}z{}_zeff.dd \n'.format(FCFCdir,output,infile,zmin,zmax,outfile[:-15],zmin,zmax))
                        f.close()
                    else:
                        pair1 = np.loadtxt('{}_z{}z{}.dd'.format(outfile[:-15],zmin,zmax),usecols=2)
                        pair2 = np.loadtxt('{}_z{}z{}_zeff.dd'.format(outfile[:-15],zmin,zmax),usecols=2)
                        zeff.append(pair2/pair1)
        if os.path.exists(output+'zeff-pair/{}_z{}z{}.dd'.format('BGS_BRIGHT_S',0.1,0.5)):            
            ZEFF['zeff-pair'] = np.array(zeff)
            ZEFF.write('DESI_SV3_neff_zeffx2_allNS.dat',format='ascii')
            ZEFF[ZEFF['GC']=='all'].write('DESI_SV3_neff_zeffx2.dat',format='ascii')

elif args.task == 'neff':
    from scipy.integrate import quad, simps
    c = 299792.458
    Om = 0.31

    def inv_efunc(z):
        return 1 / np.sqrt(1 - Om + Om * (1+z)**3)

    def cmv_dist(z):
        return quad(inv_efunc, 0, z)[0] * c / 100

    def shell_vol(area, z1, z2):
        d1 = cmv_dist(z1)
        d2 = cmv_dist(z2)
        vol = np.abs(4 * np.pi / 3. * (d2**3 - d1**3))
        full_area = 4 * np.pi * (180 / np.pi)**2
        return vol * area / full_area

    def combineNS(ifile, P0):
        zs, nbar, cnt, vol = np.loadtxt(ifile, usecols=(0,3,4,5), unpack=True)
        Veff = (nbar*P0/(nbar*P0+1))**2*vol
        nz = cnt / vol
        return [zs, nz, sum(Veff)]

    def neff(z, nz, sel):
        z = z[sel]
        nz = nz[sel]
        chi = np.array([cmv_dist(zi) for zi in z])
        neff2 = simps(nz**2 * chi**2, x=chi) / simps(chi**2, x=chi)
        return np.sqrt(neff2)

    # directory and zmin,zmax
    gals = ['LRG','ELG','BGS_ANY','QSO']+['LRG_main','ELG_HIP','BGS_BRIGHT']
    root = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/3/'
    print('gal GC zrange neff Veff')
    print('# x x x (e-4) ((Gpc/h)^3)')
    for gal in gals:
        origin = '{}{}_nz.dat'.format(root,gal)
        originN = '{}{}_N_nz.dat'.format(root,gal)
        originS = '{}{}_S_nz.dat'.format(root,gal)
        if   (gal[:3] == 'LRG'):
            zmins = [0.4,0.6,0.8,0.6]
            zmaxs = [0.6,0.8,1.1,1.1]
            P0=1e4
        elif (gal[:3] == 'ELG'):
            zmins = [0.8,1.1,0.8]
            zmaxs = [1.1,1.5,1.5]
            P0=4e3
        elif (gal[:3] == 'BGS'):
            zmins = [0.1,0.2,0.3,0.1]
            zmaxs = [0.3,0.4,0.5,0.5]
            P0=7e3
        elif gal == 'QSO':
            zmins = [0.8,1.1,1.5,0.8]
            zmaxs = [1.1,1.5,2.1,2.1]
            P0=6e3
        # calculate the neff        
        for zmin,zmax in zip(zmins,zmaxs):
            for GC,path in zip(['all','N  ','S  '],[origin,originN,originS]):
                z, nz,V_eff = combineNS(path,P0=P0)
                sel = (z >= zmin) & (z <= zmax)
                n_eff = neff(z, nz, sel)
                print('{} {} {}<z<{} {} {}'.format(gal,GC,zmin,zmax,n_eff*1e4, V_eff/1e9))
            print('\n')