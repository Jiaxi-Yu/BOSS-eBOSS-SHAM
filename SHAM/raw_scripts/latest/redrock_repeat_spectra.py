from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
from scipy.stats import norm
import os


c_kms = 299792.

plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=12)

def write_spall_redrock_join(spallname, zbestname, output):

    print('Reading spAll')
    spall = fitsio.read(spallname,
                columns=['PLATE', 'MJD', 'FIBERID', 'THING_ID',
                         'BOSS_TARGET1', 'EBOSS_TARGET0', 'EBOSS_TARGET1', 
                         'SN_MEDIAN', 
                         'SPEC1_G', 'SPEC1_R', 'SPEC1_I', 
                         'SPEC2_G', 'SPEC2_R', 'SPEC2_I',
                         'ZWARNING', 'ZWARNING_NOQSO', 'Z', 'Z_NOQSO', 'DOF', 
                         'RCHI2DIFF', 'RCHI2DIFF_NOQSO'])

    print('Reading zbest')
    redrock = fitsio.read(zbestname)
    
    print('Making tables')
    ta = Table(spall)
    tc = Table(redrock)

    tc['PLATE'], tc['MJD'], tc['FIBERID'] = \
        targetid2platemjdfiber(tc['TARGETID'])
    
    for name in tc.colnames:
        if name not in ['PLATE', 'MJD', 'FIBERID']:
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
    else:
        print("Target type unknown", target)
        return

    return spall[w]

def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return plate, mjd, fiber

 
def get_delta_velocities_from_repeats(spall,proj,target, spec1d=0, redrock=0, redmonster=0):
    
    if os.path.exists('{}-{}_deltav-z.fits.gz'.format(proj,target)):
        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'z':[], 'sn_i': [], 'sn_z': []}
        hdu = fits.open('{}-{}_deltav-z.fits.gz'.format(proj,target))
        data = hdu[1].data
        for k in info.keys():
            info[k] = np.array(data[k])
    else:
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
            

        print('Total galaxies', len(spall))
        w =  (spall[zwar_field] == 0) | (spall[zwar_field]==4)
        print(' cut on zwarn=0 or zwarn=4:', np.sum(w))
        #w &= ((spall['SN_MEDIAN'][:, 3] > 0.5) | (spall['SN_MEDIAN'][:, 4] > 0.5))
        #print(' cut on SN i-band > 0.5 or SN z-band > 0.5:', sum(w))

        spall = spall[w]

        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'z':[], 'sn_i': [], 'sn_z': []}

        uthid, index, inverse, counts = np.unique(spall['THING_ID'], 
                                                return_index=True, 
                                                return_inverse=True, 
                                                return_counts=True)

        w = (counts == 2)
        print('Selecting only duplicates', np.sum(w), w.size)
        uthid = uthid[w]

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
            dc1 = spall[chi2diff_field][j1] 
            dc2 = spall[chi2diff_field][j2] 
            if redrock==0:
                dc1 *= 1 + (spall[dof_field][j1] -1)
                dc2 *= 1 + (spall[dof_field][j2] -1)

            dv = (z1-z2)*c_kms/(1+np.min([z1, z2]))
            dc_min = np.min([dc1, dc2])
            z_min = np.min([z1, z2])
            sn_i = np.min([spall['SN_MEDIAN'][j1, 3], spall['SN_MEDIAN'][j2, 3]])
            sn_z = np.min([spall['SN_MEDIAN'][j1, 4], spall['SN_MEDIAN'][j2, 4]])
            info['thids'].append(thid) 
            info['delta_v'].append(dv)
            info['delta_chi2'].append(dc_min)
            info['z'].append(z_min)
            info['sn_i'].append(sn_i)
            info['sn_z'].append(sn_z)

        cols = []
        for k in info.keys():
            info[k] = np.array(info[k])
            cols.append(fits.Column(name=k,format=str(len(info[k]))+'D',array=info[k]))
        hdulist = fits.BinTableHDU.from_columns(cols)
        hdulist.writeto('{}-{}_deltav-z.fits.gz'.format(proj,target),overwrite=True)

    return info


def plot_deltav_deltachi2(info,dchi2_values=9, title=None, save=0):
    
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
    for i, dchi2 in enumerate(dchi2_values):
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
    plt.xlabel(r'$\Delta \chi^2$')
    plt.ylabel(r'$\Delta v$ (km/s)')
    #plt.legend(loc=0, fontsize=8)
    plt.tight_layout() 
    if title:
        plt.title(title+' %d pairs'%(npairs))
    if save:
        plt.savefig(save, bbox_inches='tight')

    print('Total pairs in plot', dc.size)

def plot_deltav_hist(info, bins=300, zmin=0.6, zmax=1.0, max_dv=500., min_deltachi2=9, title=None, save=0):

    dc = info['delta_chi2']
    dv = info['delta_v']
    z = info['z']

    #-- select good redshifts, inside redshift range, reject outliers
    w = (dc > min_deltachi2)&(z>=zmin)&(z<=zmax)&(abs(dv)<max_dv)
    wfit = w&(abs(dv)< 250)
    
    plt.figure(figsize=(5,4))
    

    bins = np.linspace(-max_dv, max_dv, bins)
    _ = plt.hist(dv[w], bins=bins, histtype='step', density=True)
    
    #-- fit a Gaussian
    (mu, sigma) = norm.fit(dv[wfit])
    gauss = 1/np.sqrt(2*np.pi)/sigma*np.exp( -(bins-mu)**2/(2*sigma**2))

    plt.plot(bins, gauss, label=r'Gaussian fit $\mu = {mu:.1f}, \ \sigma = {sigma:.1f}$'.format(mu=mu, sigma=sigma))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('Normalized distribution')
    plt.legend(loc=0)
    if title:
        plt.title(title+r' %d pairs with $\Delta \chi^2 > %.1f$'%(dv[w].size, min_deltachi2))
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_all_deltav_deltachi2(spall,proj,dchi2=9):

    spall = Table.read(spall)

    #for target in ['ELG', 'LRG']:
    for target in ['LRG']:
        sp = get_targets(spall, target=target)

        #- spec1d
        #info0 = get_delta_velocities_from_repeats(sp, spec1d=1)
        #plot_deltav_deltachi2(info0, title=f'eBOSS {target} repeats - spec1d', 
        #           save=f'plots/eboss-{target}-repeats-spec1d.pdf')
        
        #-- Redrock
        info1 = get_delta_velocities_from_repeats(sp,proj,target, spec1d=1)
        
        plot_deltav_deltachi2(info1, dchi2_values=dchi2,title='eBOSS {} repeats - redrock'.format(target), 
                   save='eboss-{}-repeats-spec1d.pdf'.format(target))

def plot_all_deltav_histograms(spall,proj,dchi2=9,maxdv=500):

    spall = Table.read(spall)

    for target in ['LRG']:
        sp = get_targets(spall, target=target)
        info = get_delta_velocities_from_repeats(sp,proj,target,spec1d=1)
        plot_deltav_hist(info, min_deltachi2=dchi2,  max_dv=maxdv,title='eBOSS {} -'.format(target), 
                   save='eboss-{}-repeats-spec1d-histogram.pdf'.format(target))
        

def get_false_positives(spall, redrock=0, spec1d=0):

    spall = Table.read(spall)

    if spec1d:
        zwar_field = 'ZWARNING'
        chi2diff_field = 'RCHI2DIFF'
        z_field = 'Z'
        dof_field = 'DOF'
        zwar = spall[zwar_field] - 1
    elif redrock:
        zwar_field = 'ZWARN_REDROCK'
        chi2diff_field = 'DELTACHI2_REDROCK'
        z_field = 'Z_REDROCK'
        zwar = spall[zwar_field] 
    

    print('Total skies', len(spall))
    w =  (zwar == 0) | (zwar == 4) 
    print(' cut on zwarn=0 or zwarn=4:', np.sum(w))

    ww = (zwar == 0 ) 
    print(' Confident detections in sky fibers', np.sum(ww), np.sum(ww)/np.sum(w))

#write_spall_redrock_join('/uufs/chpc.utah.edu/common/home/sdss/ebosswork/eboss/spectro/redux/v5_13_0/spAll-v5_13_0.fits', 
#                         '/uufs/chpc.utah.edu/common/home/bolton-group1/bolton_data2/kdawson/bautista/redrock_redux/v5_13_0/zbest-v5_13_0.fits',
#                         'data/spAll-zbest-v5_13_0.fits')
#write_spall_repeats('spAll-v5_13_0.fits', 'spAll-zbest-v5_13_0-repeats-2x.fits')
#plot_all_deltav_deltachi2('spAll-zbest-v5_13_0-repeats-2x.fits','eBOSS',dchi2=25)
plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x.fits','eBOSS',dchi2=25)
#write_spall_redrock_join('spAll-zbest-v5_13_0-repeats-2x.fits', 
#                         '/uufs/chpc.utah.edu/common/home/bolton-group1/bolton_data2/kdawson/bautista/redrock_redux/v5_13_0_no_andmask/zbest-v5_13_0_no_andmask-eBOSS.fits',
#                         'spAll-zbest-v5_13_0_no_andmask-repeats-2x-eBOSS.fits')
#plot_all_deltav_deltachi2('spAll-zbest-v5_13_0_no_andmask-repeats-2x-eBOSS.fits')
#plot_all_deltav_histograms('spAll-zbest-v5_13_0_no_andmask-repeats-2x-eBOSS.fits')
#plt.show()
#get_false_positives('spAll-zbest-v5_13_0_no_andmask-skies.fits', spec1d=1)
#get_false_positives('spAll-zbest-v5_13_0_no_andmask-skies.fits', redrock=1)

