from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt
import os
"""
from lmfit.models import VoigtModel
mod = VoigtModel()
from lmfit.models import LorentzianModel
mod = LorentzianModel()
pars = mod.guess(dens, x=BIN)
out = mod.fit(dens, pars, x=BIN)
print(out.fit_report(min_correl=0.25))
"""

c_kms = 299792.
home = '/global/homes/j/jiaxi/'

plt.rc('text', usetex=False)
plt.rc('font', family='serif', size=12)

def gaussian(x,a,sigma,mu):
    return a/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2*sigma**2))
def lorentzian(x,a,w,p):#
    return a/np.pi/w/(1+((x-p)*2/w)**2)

def targetid2platemjdfiber(targetid):
    fiber = targetid % 10000
    mjd = (targetid // 10000) % 100000
    plate = (targetid // (10000 * 100000))
    return plate, mjd, fiber

def write_spall_redrock_join(spallname, zbestname, output):

    print('Reading spAll')
    spall = fitsio.read(spallname,
                columns=['PLATzE', 'MJD', 'FIBERID', 'THING_ID',
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
        zerr_field = 'ZERR_REDROCK'
    elif redmonster:
        zwar_field = 'ZWARNING_REDMONSTER'
        chi2diff_field = 'RCHI2DIFF_REDMONSTER'
        z_field = 'Z_REDMONSTER'
        dof_field = 'DOF_REDMONSTER'
    zerr_field = 'ZERR_REDROCK'
    # select repeats
    if os.path.exists('{}Vsmear/{}-{}_deltav_z{}z{}.fits.gz'.format(home,proj,target,zmin,zmax)):
        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'z':[],'zerr':[], 'sn_i': [], 'sn_z': []}
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

        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], 'zerr':[], 'z':[], 'sn_i': [], 'sn_z': []}

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

            dv = (z1-z2)*c_kms/(1+z_clustering)
            dc_min = np.min([dc1, dc2])
            sn_i = np.min([spall['SN_MEDIAN'][j1, 3], spall['SN_MEDIAN'][j2, 3]])
            sn_z = np.min([spall['SN_MEDIAN'][j1, 4], spall['SN_MEDIAN'][j2, 4]])
            info['zerr'].append(np.sqrt(spall[zerr_field][j1]**2+spall[zerr_field][j2]**2)*c_kms/(1+z_clustering))
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

def jacknife_hist(dvsel,bins,nsub,save=0,gaussian=True):
    if gaussian:
        if os.path.exists(save):
            hists = np.loadtxt(save)
            output = [hists[:,0],hists[:,1:]]
        else:
            # -- generate catalogues
            BIN = (bins[1:]+bins[:-1])/2
            partlen = len(dvsel)//nsub
            dvsub = [i for i in range(nsub)]
            dens = [i for i in range(nsub)]
            stds = []
            for index in range(nsub):
                if index == nsub-1:
                    dvsub[index] = dvsel[:index*partlen]
                else:
                    dvsub[index] = np.append(dvsel[:index*partlen],dvsel[(index+1)*partlen:])

                dens[index],BINS = np.histogram(dvsub[index], bins=bins)

            np.savetxt(save,np.hstack((BIN.reshape(len(BIN),1),np.array(dens).T)),header='bins(width=1) density for 100 sub-catalogues')

            output  = [BIN,np.array(dens).T]
    else:
        # -- generate catalogues
        partlen = len(dvsel)//nsub
        dvsub = [i for i in range(nsub)]
        dens = [i for i in range(nsub)]
        for index in range(nsub):
            if index == nsub-1:
                dvsub[index] = dvsel[:index*partlen]
            else:
                dvsub[index] = np.append(dvsel[:index*partlen],dvsel[(index+1)*partlen:])

            dens[index] = np.std(dvsub[index])
        output = np.std(dens)
    
    return output
    
def plot_deltav_hist(info,target,zrange,max_dv=500., min_deltachi2=9, nsubvolume = 1000,title=None, save=0):
    #-- select inside redshift range, reject outliers
    dc = info['delta_chi2']
    dv = info['delta_v']
    z = info['z']
    zerr = info['zerr']
    w = (dc > min_deltachi2)&(abs(dv)<max_dv)
    dvsel = dv[w]
    
    # binning dv[w]
    binwidth = 5
    bins = np.arange(-max_dv, max_dv+1, binwidth)
    outliern = len(dv[(dv>-1000)&(dv<-max_dv)])
    outlierp = len(dv[(dv<1000)&(dv>max_dv)])
    dens,BINS = np.histogram(dvsel,bins=bins)
    norm = np.sum(dens)
    dens = dens/norm
    
    # histogram jacknife
    BIN,hists = jacknife_hist(dvsel,bins,nsub = nsubvolume,save = save[:29]+target+'-'+zrange+'-maxdv'+str(max_dv)+'-jacknife.dat')
    hists =hists/norm
    histstd = np.std(hists,axis=1)*np.sqrt(nsubvolume)
    histcovR = np.linalg.pinv(np.cov(hists)*nsubvolume)*(hists.shape[1]-hists.shape[0]-2)/(hists.shape[1]-1)
    
    #-- fit a Gaussian: delta_v
    popt, pcov = curve_fit(gaussian,BIN,dens,sigma=histstd)
    res = gaussian(BIN,*popt)-dens
    popt1, pcov1 = curve_fit(lorentzian,BIN,dens,sigma=histstd)
    res1 = lorentzian(BIN,*popt1)-dens
    
    # directly calculate the std
    STD = jacknife_hist(dv[abs(dv)<1000],bins,nsub = nsubvolume,gaussian=False)
    print('std calculation: Vsmear = [{:.1f},{:.1f}]'.format(np.std(dv[abs(dv)<1000])-STD*np.sqrt(nsubvolume),np.std(dv[abs(dv)<1000])+STD*np.sqrt(nsubvolume)))
    print('Gaussian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,popt[1]-np.sqrt(np.diag(pcov))[1],popt[1]+np.sqrt(np.diag(pcov))[1]))    
    print('Lorentzian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,(popt1[1]-np.sqrt(np.diag(pcov1))[1])/2/np.sqrt(2*np.log(2)),(popt1[1]+np.sqrt(np.diag(pcov1))[1])/2/np.sqrt(2*np.log(2))))    
        
    # plot the gaussian: delta_v
    plt.figure(figsize=(8,6))
    plt.errorbar(BIN,dens,histstd,color='k', marker='o',ecolor='k',ls="none")
    plt.scatter(-max_dv,outliern/norm,c='r')
    plt.scatter(max_dv,outlierp/norm,c='r')
    plt.plot(BIN, gaussian(BIN,*popt), label=r'Gaussian fit $\sigma = {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(popt[1],np.sqrt(np.diag(pcov))[1],np.sqrt(np.diag(pcov))[1],res.dot(histcovR.dot(res)),len(res)))
    plt.plot(BIN, lorentzian(BIN,*popt1), label='Lorentzian fit '+r'$\frac{w}{2\sqrt{2ln2}}$'+'$= {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$,$\chi^2$ /dof = {3:.1f}/{4:}'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2)),res1.dot(histcovR.dot(res1)),len(res1)))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('counts')
    plt.legend(loc=1)
    #plt.yscale('log')
    plt.ylim(0,max(dens)*1.3)

    if title:
        plt.title(title+' with {} pairs, std = {:.1f} $\pm$ {:.1f}, fitting by curve_fit'.format(dvsel.size,np.std(dv[abs(dv)<1000]),STD*np.sqrt(nsubvolume)))
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.close()
        
    #-- fit a Gaussian: zerr
    ratios = np.linspace(-5,5,101)
    ratio = (ratios[1:]+ratios[:-1])/2
    ratiodens,ratiobin = np.histogram(dv/zerr,ratios)
    ratiodens = ratiodens/len(ratiodens)
    
    BIN,hists = jacknife_hist(dv/zerr,ratios,nsub = nsubvolume,save = save[:29]+target+'-maxdv'+str(max_dv)+'-jacknife-zerr.dat')
    hists =hists/len(ratiodens)
    histstd = np.std(hists,axis=1)*np.sqrt(nsubvolume)
    histcovR = np.linalg.pinv(np.cov(hists)*nsubvolume)*(hists.shape[1]-hists.shape[0]-2)/(hists.shape[1]-1)
    
    popt2, pcov2 = curve_fit(gaussian,ratio,ratiodens,sigma=histstd)
    res2 = gaussian(ratio,*popt2)-ratiodens
         
    plt.figure(figsize=(8,6))
    plt.scatter(ratio,ratiodens)
    plt.errorbar(ratio,ratiodens,histstd,color='k', marker='o',ecolor='k',ls="none")
    plt.plot(ratio, gaussian(ratio,*popt2), label=r'Gaussian fit $\sigma = {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(popt2[1],np.sqrt(np.diag(pcov2))[1],np.sqrt(np.diag(pcov2))[1],res2.dot(histcovR.dot(res2)),len(res2)))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('normalised counts')
    plt.legend(loc=1)
    if title:
        plt.title(title+' repetitive samples: $\Delta$ v v.s. zerr')
        #plt.title(title+' with {} pairs, std = {:.1f} $\pm$ {:.1f}, fitting by curve_fit'.format(dv[w].size,np.std(dv[abs(dv)<1000]),STD*np.sqrt(nsubvolume)))
    plt.tight_layout()
    if save:
        plt.savefig(save[:-4]+'-zerr.png', bbox_inches='tight')
    plt.close()
    
def lnprior(par):
    a, fwhm,maxpos = par
    if 0 <= a <= 20 and 0 <= fwhm <= 160 and -10 <= maxpos <= 10:
        return 0.0
    return -np.inf  

def plot_deltav_hist_emcee(info,target,zrange,max_dv=500., min_deltachi2=9, nsubvolume = 1000,title=None, save=0):
    #-- select inside redshift range, reject outliers
    dc = info['delta_chi2']
    dv = info['delta_v']
    z = info['z']
    zerr = info['zerr']
    w = (dc > min_deltachi2)&(abs(dv)<max_dv)
    dvsel = dv[w]
    
    # binning dv[w]
    binwidth = 5
    bins = np.arange(-max_dv, max_dv+1, binwidth)
    outliern = len(dv[(dv>-1000)&(dv<-max_dv)])
    outlierp = len(dv[(dv<1000)&(dv>max_dv)])
    dens,BINS = np.histogram(dvsel,bins=bins)
    norm = np.sum(dens)
    dens = dens/norm
    
    # histogram jacknife
    BIN,hists = jacknife_hist(dvsel,bins,nsub = nsubvolume,save = save[:29]+target+'-maxdv'+str(max_dv)+'-jacknife.dat')
    hists =hists/norm
    histstd = np.std(hists,axis=1)*np.sqrt(nsubvolume)
    histcovR = np.linalg.pinv(np.cov(hists)*nsubvolume)*(hists.shape[1]-hists.shape[0]-2)/(hists.shape[1]-1)
    
    def lnprob_gaussian(par):
        lp = lnprior(par)
        if not np.isfinite(lp):
            return -np.inf
        resG = gaussian(BIN,*par)-dens
        return lp - 0.5 * resG.dot(histcovR.dot(resG))

    def lnprob_lorentzian(par):
        lp = lnprior(par)
        if not np.isfinite(lp):
            return -np.inf
        resL = lorentzian(BIN,*par)-dens
        return lp - 0.5 * resL.dot(histcovR.dot(resL))

    # emcee fitting
    import emcee
    ndim, nwalkers = 3, 100
    ini = np.array([1, 50., 0.])
    ini = [ini + 1e-4 * np.random.randn(ndim) \
            for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, \
            lnprob_gaussian)    
    sampler.run_mcmc(ini, 500)
    samples = sampler.chain[:, 100:, :].reshape((-1, ndim))
    a,sigma,mu = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
                    zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    res = gaussian(BIN,a[0],sigma[0],mu[0])-dens
    # lorentzian
    ini = np.array([0.1, 50., 0.])
    ini = [ini + 1e-4 * np.random.randn(ndim) \
            for i in range(nwalkers)]
    sampler1 = emcee.EnsembleSampler(nwalkers, ndim, \
            lnprob_lorentzian)
    sampler1.run_mcmc(ini, 500)
    samples1 = sampler1.chain[:, 100:, :].reshape((-1, ndim))
    a1,w,p = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
                    zip(*np.percentile(samples1, [16, 50, 84], axis=0)))
    res1 = lorentzian(BIN,a1[0],w[0],p[0])-dens

    # directly calculate the std
    STD = jacknife_hist(dv[abs(dv)<1000],bins,nsub = nsubvolume,gaussian=False)
    print('std calculation: Vsmear = [{:.1f},{:.1f}]'.format(np.std(dv[abs(dv)<1000])-STD*np.sqrt(nsubvolume),np.std(dv[abs(dv)<1000])+STD*np.sqrt(nsubvolume)))
    print('Gaussian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,sigma[0]-sigma[1],sigma[0]+sigma[2]))
    print('Lorentzian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,(w[0]-w[1])/2/np.sqrt(2*np.log(2)),(w[0]+w[2])/2/np.sqrt(2*np.log(2))))

    # plot the posterior
    import corner
    fig = corner.corner(samples, labels=[r'a', r'$\sigma$', r'$\mu$'])
    plt.savefig(save[:-4]+'_posteriorG.png')
    plt.close()
    fig = corner.corner(samples1, labels=[r'a', r'w', r'$p_0$'])
    plt.savefig(save[:-4]+'_posteriorL.png')
    plt.close()
    # plot the Delta v vs models
    plt.figure(figsize=(8,6))
    plt.errorbar(BIN,dens,histstd,color='k', marker='o',ecolor='k',ls="none")
    plt.scatter(-max_dv,outliern/norm,c='r')
    plt.scatter(max_dv,outlierp/norm,c='r')
    plt.plot(BIN, gaussian(BIN,a[0],sigma[0],mu[0]), label=r'Gaussian fit $\sigma = {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(sigma[0],sigma[1],sigma[2],res.dot(histcovR.dot(res)),len(res)))
    plt.plot(BIN, lorentzian(BIN,a1[0],w[0],p[0]),label='Lorentzian fit '+r'$\frac{w}{2\sqrt{2ln2}}$'+'$= {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(w[0]/2/np.sqrt(2*np.log(2)),w[1]/2/np.sqrt(2*np.log(2)),w[2]/2/np.sqrt(2*np.log(2)),res1.dot(histcovR.dot(res1)),len(res1)))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('normalised counts')
    plt.legend(loc=1)
    #plt.yscale('log')
    plt.ylim(0,max(dens)*1.3)
    if title:
        plt.title(title+' with {} pairs, std = {:.1f} $\pm$ {:.1f}, fitting by emcee'.format(dvsel.size,np.std(dv[abs(dv)<1000]),STD*np.sqrt(nsubvolume)))
    plt.tight_layout()
    if save:
        plt.savefig(save[:-4]+'_emcee.png', bbox_inches='tight')
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

    plot_deltav_hist(info,target,zrange='z{}z{}'.format(zmin,zmax),min_deltachi2=dchi2,  max_dv=maxdv,title='{} {} {}<z<{}'.format(proj,target,zmin,zmax), save='{}Vsmear/{}-{}-repeats-{}-dchi2_{}-z{}z{}.png'.format(home,proj,target,zsource,dchi2,zmin,zmax))
    plot_deltav_hist_emcee(info,target,zrange='z{}z{}'.format(zmin,zmax),min_deltachi2=dchi2,  max_dv=maxdv,title='{} {} {}<z<{}'.format(proj,target,zmin,zmax), save='{}Vsmear/{}-{}-repeats-{}-dchi2_{}-z{}z{}.png'.format(home,proj,target,zsource,dchi2,zmin,zmax))
        

# eBOSS LRG:
#write_spall_redrock_join('spAll-v5_13_0.fits', 'spAll_trimmed_pREDROCK.fits','spAll-zbest-v5_13_0.fits')
#write_spall_repeats('spAll-zbest-v5_13_0.fits', 'spAll-zbest-v5_13_0-repeats-2x_redrock.fits')

#plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','BOSS',zmin=0.2,zmax=0.43,target='LOWZ',dchi2=9,spec1d=1,maxdv=140)

zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
maxdvs = [235,275,275,300,255,360]
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

#########################################################################################    
