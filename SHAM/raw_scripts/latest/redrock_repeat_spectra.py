from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import os


c_kms = 299792.
home = '/global/homes/j/jiaxi/'

plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=12)

def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return plate, mjd, fiber

def write_spall_redrock_join(spallname, zbestname, output):

    print('Reading spAll')
    spall = fitsio.read(spallname,
                columns=['PLATE', 'MJD', 'FIBERID', 'THING_ID',
                         'BOSS_TARGET1', 'EBOSS_TARGET0', 'EBOSS_TARGET1', 
                         'SN_MEDIAN', 
                         'SPEC1_G', 'SPEC1_R', 'SPEC1_I', 
                         'SPEC2_G', 'SPEC2_R', 'SPEC2_I',
                         'CHUNK','THING_ID_TARGETING',
                         'ZWARNING', 'ZWARNING_NOQSO', 'Z', 'Z_NOQSO', 'DOF', 
                         'RCHI2DIFF', 'RCHI2DIFF_NOQSO', 'SPECPRIMARY'])

    print('Reading zbest')
    redrock = fitsio.read(zbestname)
    
    print('Making tables')
    ta = Table(spall)
    tc = Table(redrock)

    tc['PLATE'], tc['MJD'], tc['FIBERID'] = \
        targetid2platemjdfiber(tc['TARGETID'])
    
    for name in tc.colnames:
        if name not in ['PLATE', 'MJD', 'FIBERID']:
            if name=='Z':
                tc[name].name = 'Z-_REDROCK'
            elif name == 'Z_REDROCK':
                pass
            else:
                tc[name].name = name+'_REDROCK'

    print('Joining tables')
    tac = join(ta, tc, keys=['PLATE', 'MJD', 'FIBERID'], 
               join_type='left')

    print('Writing joined table')
    tac.write(output, format='fits', overwrite=True)

def write_spall_repeats(spallin, spallout, ncount=2):

    a = Table.read(spallin)

    thid = a['THING_ID']

    #-- get unique values and number of occurrencies
    uthid, counts = np.unique(thid, return_counts=True) 

    #-- repeats are entries with more than one count
    if ncount is None:
        ruthid = uthid[counts>1]
    else:
        ruthid = uthid[counts==ncount]

    #-- get elements that are repeated
    w = np.isin(thid, ruthid) 
 
    print(sum(w), 'repeats of ', w.size)
    t = Table(a[w])
    t.write(spallout, overwrite=True)


def get_targets(spall, target='LRG'):

    if target=='LRG':
        w = (spall['EBOSS_TARGET0'] & 2**2 > 0)|\
            (spall['EBOSS_TARGET1'] & 2**1 > 0)|\
            (spall['EBOSS_TARGET1'] & 2**2 > 0)
    elif target=='ELG':
        w = (spall['EBOSS_TARGET1'] & 2**43 > 0)|\
            (spall['EBOSS_TARGET1'] & 2**44 > 0) 
    elif target=='CMASS':
        w = spall['BOSS_TARGET1'] & 2**1 > 0
    elif target=='LOWZ':
        w = spall['BOSS_TARGET1'] & 2**0 > 0
    elif target == 'BOSS':
        w = (spall['BOSS_TARGET1'] & 2**0 > 0)|\
            (spall['BOSS_TARGET1'] & 2**1 > 0)
    else:
        print("Target type should be: ELG, LRG, CMASS, LOWZ, BOSS")
        return

    return spall[w]

def get_delta_velocities_from_repeats(spall,proj,target,zmin,zmax,spec1d=0, redrock=0, redmonster=0):
    # zwarning, chi2difference
    if spec1d:
        zwar_field = 'ZWARNING_NOQSO'
        chi2diff_field = 'RCHI2DIFF_NOQSO'
        z_field = 'Z_NOQSO'
        dof_field = 'DOF'
    elif redrock:
        zwar_field = 'ZWARN_REDROCK'
        chi2diff_field = 'DELTACHI2_REDROCK'
        z_field = 'Z_REDROCK'
    elif redmonster:
        zwar_field = 'ZWARNING_REDMONSTER'
        chi2diff_field = 'RCHI2DIFF_REDMONSTER'
        z_field = 'Z_REDMONSTER'
        dof_field = 'DOF_REDMONSTER'
    print(zwar_field, ',',z_field)    
    # select repeats
    if os.path.exists('{}Vsmear/{}-{}_deltav_z{}z{}.fits.gz'.format(home,proj,target,zmin,zmax)):
        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'z':[], 'sn_i': [], 'sn_z': []}
        hdu = fits.open('{}Vsmear/{}-{}_deltav_z{}z{}.fits.gz'.format(home,proj,target,zmin,zmax))
        data = hdu[1].data
        hdu.close()
        for k in info.keys():
            info[k] = np.array(data[k])
        print('{}<z<{} has {} duplicates'.format(zmin,zmax,len(np.array(data['z']))))

    else:
        print('Total galaxies', len(spall))
        w =  (spall[zwar_field] == 0) | (spall[zwar_field]==4)
        print(' cut on zwarn=0 or zwarn=4:', np.sum(w))
        #w &= ((spall['SN_MEDIAN'][:, 3] > 0.5) | (spall['SN_MEDIAN'][:, 4] > 0.5))
        #print(' cut on SN i-band > 0.5 or SN z-band > 0.5:', sum(w))

        spall = spall[w]

        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'z':[], 'sn_i': [], 'sn_z': []}

        uthid, index, inverse, counts = np.unique(spall['THING_ID'], return_index=True, return_inverse=True, return_counts=True)

        w = (counts == 2)
        print('Selecting only duplicates', np.sum(w), w.size)
        uthid = uthid[w]
        zflag=[]

        for thid in uthid:
            if thid in info['thids']:
                continue
            w = np.where(spall['THING_ID'] == thid)[0]
            if len(w) == 0:
                print(thid)

            if spall[chi2diff_field][w[0]] < spall[chi2diff_field][w[1]]:
                j1 = w[1]
                j2 = w[0]
            else:
                j1 = w[0]
                j2 = w[1]

            z1 = spall[z_field][j1]
            z2 = spall[z_field][j2]
            z_clustering = spall["SPECPRIMARY"][j2]*z2+spall["SPECPRIMARY"][j1]*z1
            dc1 = spall[chi2diff_field][j1] 
            dc2 = spall[chi2diff_field][j2] 
            if redrock==0:
                dc1 *= 1 + (spall[dof_field][j1] -1)
                dc2 *= 1 + (spall[dof_field][j2] -1)

            dv = (z1-z2)*c_kms/(1+np.min([z1, z2]))
            dc_min = np.min([dc1, dc2])
            sn_i = np.min([spall['SN_MEDIAN'][j1, 3], spall['SN_MEDIAN'][j2, 3]])
            sn_z = np.min([spall['SN_MEDIAN'][j1, 4], spall['SN_MEDIAN'][j2, 4]])
            info['thids'].append(thid) 
            info['delta_v'].append(dv)
            info['delta_chi2'].append(dc_min)
            info['z'].append(z_clustering)
            info['sn_i'].append(sn_i)
            info['sn_z'].append(sn_z)
        zflag = (np.array(info['z'])>zmin)&(np.array(info['z'])<zmax)
        print('{}<z<{} has {} duplicates'.format(zmin,zmax,len(np.array(info['z'])[zflag])))
        
        cols = []
        print('before:',np.array(info['z']).shape)
        for k in info.keys():
            info[k] = np.array(info[k])[zflag]
            cols.append(fits.Column(name=k,format='D',array=info[k]))
        hdulist = fits.BinTableHDU.from_columns(cols)
        hdulist.writeto('{}Vsmear/{}-{}_deltav_z{}z{}.fits.gz'.format(home,proj,target,zmin,zmax),overwrite=True)
        print('after:',np.array(info['z']).shape)

    return info

def jacknife_hist(dvsel,bins,nsub,max_dv,save):
    if os.path.exists(save):
        hists = np.loadtxt(save)
        return [hists[:,0],hists[:,1:]]
    else:
        # -- generate catalogues
        BIN = (bins[1:]+bins[:-1])/2
        partlen = len(dvsel)//nsub
        dvsub = [i for i in range(nsub)]
        dens = [i for i in range(nsub)]
        for index in range(nsub):
            if index == nsub-1:
                dvsub[index] = dvsel[:index*partlen]
            else:
                dvsub[index] = np.append(dvsel[:index*partlen],dvsel[(index+1)*partlen:])

            dens[index],BINS,plot = plt.hist(dvsub[index], bins=bins, histtype='step')#, density=True)
        
        np.savetxt(save,np.hstack((BIN.reshape(len(BIN),1),np.array(dens).T)),header='bins(width=1) density for 100 sub-catalogues')

    return [BIN,np.array(dens).T]
    
def plot_deltav_hist(info,max_dv=500., min_deltachi2=9, nsubvolume = 100,title=None, save=0):
    #-- select inside redshift range, reject outliers
    dc = info['delta_chi2']
    dv = info['delta_v']
    z = info['z']
    w = (dc > min_deltachi2)&(abs(dv)<max_dv)
    dvsel = dv[w]
    
    # binning dv[w]
    bins = np.arange(-max_dv, max_dv+1, 5)
    dens,BINS = np.histogram(dvsel,bins=bins)
    
    # histogram jacknife
    BIN,hists = jacknife_hist(dvsel,bins,nsub = nsubvolume,max_dv=max_dv,save = save[:40]+save[-23:-14]+'-maxdv'+str(max_dv)+'-jacknife.dat')
    histstd = np.std(hists,axis=1)*np.sqrt(nsubvolume)
    histcovR = np.linalg.pinv(np.cov(hists*nsubvolume))*(hists.shape[1]-hists.shape[0]-2)/(hists.shape[1]-1)
    
    #-- fit a Gaussian
    def gaussian(x,a,sigma,mu):
        return a/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2*sigma**2))

    popt, pcov = curve_fit(gaussian,BIN,dens,sigma=histstd)
    res = gaussian(BIN,*popt)-dens
    
    plt.figure(figsize=(7.5,6))
    plt.errorbar(BIN,dens,histstd,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(BIN, gaussian(BIN,*popt), label=r'Gaussian fit $\mu = {:.1f} \pm {:.1f}, \ \sigma = {:.1f} \pm {:.1f}$'.format(popt[2],np.sqrt(np.diag(pcov))[2], popt[1],np.sqrt(np.diag(pcov))[1]))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('counts')
    plt.legend(loc=1)
    plt.ylim(-20,max(dens)*1.3)
    if title:
        plt.title(title+r' %d pairs with $\Delta \chi^2 > %.1f$'%(dv[w].size, min_deltachi2))
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.close()

def plot_all_deltav_histograms(spall,proj,zmin,zmax,target='LRG',dchi2=9,maxdv=500,spec1d=0, redrock=0, redmonster=0):

    spall = Table.read(spall)
    sp = get_targets(spall, target=target)
    
    if spec1d:
        zsource = 'spec1d'
        info = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,spec1d=1)
    elif redrock:
        zsource = 'redrock'
        info = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,redrock=1)
    elif redmonster:
        zsource='redmonster'
        info = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,redmonster=1)

    plot_deltav_hist(info,min_deltachi2=dchi2,  max_dv=maxdv,title='eBOSS {} -'.format(target), save='{}Vsmear/{}-{}-repeats-{}-dchi2_{}-z{}z{}-histogram.png'.format(home,proj,target,zsource,dchi2,zmin,zmax))
        

# eBOSS LRG:
#write_spall_redrock_join('spAll-v5_13_0.fits', 'spAll_trimmed_pREDROCK.fits','spAll-zbest-v5_13_0.fits')
#write_spall_repeats('spAll-zbest-v5_13_0.fits', 'spAll-zbest-v5_13_0-repeats-2x_redrock.fits')

zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
maxdvs = [235,275,275,300,255,380]
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    #plot_all_deltav_deltachi2('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','eBOSS',zmin,zmax,target='LRG',dchi2=9,redrock=1)
    plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','eBOSS',zmin,zmax,target='LRG',dchi2=9,redrock=1,maxdv=maxdv)

#write_spall_repeats('spAll-v5_4_45.fits', 'spAll-zbest-v5_4_45-repeats-2x.fits')

zmins = [0.43,0.51,0.57,0.43]
zmaxs = [0.51,0.57,0.7,0.7]
maxdvs = [205,200,235,270]
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    #plot_all_deltav_deltachi2('spAll-zbest-v5_4_45-repeats-2x.fits','BOSS',zmin,zmax,target='CMASS',dchi2=9,spec1d=1)
    plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','BOSS',zmin,zmax,target='CMASS',dchi2=9,spec1d=1,maxdv=maxdv)

zmins = [0.2, 0.33,0.2]
zmaxs = [0.33,0.43,0.43]
maxdvs = [105,140,140]
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    #plot_all_deltav_deltachi2('spAll-zbest-v5_4_45-repeats-2x.fits','BOSS',zmin,zmax,target='LOWZ',dchi2=9,spec1d=1)
    plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','BOSS',zmin,zmax,target='LOWZ',dchi2=9,spec1d=1,maxdv=maxdv)

##############################################################################################
def plot_deltav_deltachi2(info,dchi2=9, title=None, save=0):
    
    dc = info['delta_chi2']
    dv = info['delta_v']
    sn_i = info['sn_i']
    sn_z = info['sn_z']
    npairs = len(dc)

    plt.figure(figsize=(5, 4))
    plt.plot( dc, dv, 'k.', ms=2, alpha=0.3)
    w = (sn_i>0.5)|(sn_z> 0.5)
    print('Cut in S/N: ', np.sum(w), w.size)
    plt.plot( dc[~w], dv[~w], 'r.', ms=2)
    dc = dc[w]
    dv = dv[w]
    plt.ylim(0.01, 1e6)
    plt.xlim(1e-1, 1e4)
    plt.xscale('log')
    plt.yscale('log')
    ylim = plt.ylim()
    plt.axhline(1000, color='r', ls='--')
    #dchi2_values = [1, 4, 9, 16, 25]
    #for i, dchi2 in enumerate(dchi2_values):
    i=0
    nspec = dc.size
    nconf = np.sum(dc>dchi2)
    ncata = np.sum((dc>dchi2)&(dv>1000))
    conf_rate = nconf/nspec
    catastrophic_rate = ncata/nconf

    print(f'N = {nspec}')
    print(f'N(delta_chi2 > {dchi2}) = {nconf}')
    print(f'N(delta_chi2 > {dchi2} & delta_v > 1000) = {ncata}')
    label = r'$\Delta \chi^2_{\rm thres} = %d, f_{\rm good}= %.2f, f_{\Delta v>1000{\rm km/s}} = %.3f$'%\
              (dchi2, conf_rate, catastrophic_rate)
    plt.axvline(dchi2, ls=':', color='C%d'%i, label=label)
    ####################################
    plt.xlabel(r'$\Delta \chi^2$')
    plt.ylabel(r'$\Delta v$ (km/s)')
    #plt.legend(loc=0, fontsize=8)
    plt.tight_layout() 
    if title:
        plt.title(title+' %d pairs'%(npairs))
    if save:
        plt.savefig(save, bbox_inches='tight')

    print('Total pairs in plot', dc.size)
    
def plot_all_deltav_deltachi2(spall,proj,zmin,zmax,target='LRG',dchi2=9,spec1d=0, redrock=0, redmonster=0):

    spall = Table.read(spall)
    sp = get_targets(spall, target=target)

    #- read repetitive catalogues for targets
    if spec1d:
        zsource = 'spec1d'
        info1 = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,spec1d=1)
    elif redrock:
        zsource = 'redrock'
        info1 = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,redrock=1)
    elif redmonster:
        zsource='redmonster'
        info1 = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,redmonster=1)
        
    plot_deltav_deltachi2(info1,dchi2,title='eBOSS {} repeats - redrock'.format(target), 
               save='{}Vsmear/{}-{}-repeats-{}-dchi2_{}-z{}z{}.pdf'.format(home,proj,target,zsource,dchi2,zmin,zmax))

