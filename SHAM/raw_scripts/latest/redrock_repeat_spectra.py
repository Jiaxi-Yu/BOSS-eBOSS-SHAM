#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
from scipy.optimize import curve_fit
import pylab as plt
import os
from lmfit.models import PseudoVoigtModel
"""
import numpy.ma as ma
test = ma.array(tac['WEIGHT_SPEC']) 
tac['WEIGHT_SPEC'][np.where(test.mask==True)]=1
"""
c_kms = 299792.
home = '/global/homes/j/jiaxi/Vsmear-photo'
cutind = 35


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
                columns=['PLATE', 'MJD', 'FIBERID', 'THING_ID',
                         'BOSS_TARGET1', 'EBOSS_TARGET0', 'EBOSS_TARGET1', 
                         'SN_MEDIAN', 
                         'OBJID',
                         'CHUNK','THING_ID_TARGETING',
                         'ZWARNING', 'ZWARNING_NOQSO', 'Z', 'Z_NOQSO', 'DOF', 
                         'RCHI2DIFF', 'RCHI2DIFF_NOQSO', 'SPECPRIMARY'])

    print('Reading zbest')
    redrock = fitsio.read(zbestname)
    
    print('Making tables')
    ta = Table(spall)
    tc = Table(redrock)
    redrock=[];spall=[]
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
    if output.find('photo')==-1:
        tac = join(ta, tc, keys=['PLATE', 'MJD', 'FIBERID'], 
                   join_type='left')
    else:
        tac_tmp = join(ta, tc, keys=['PLATE', 'MJD', 'FIBERID'], 
                   join_type='left')
        ta = [];tc=[];
        print('reading photo-info file')
        photo_tmp =  fitsio.read('Photo_dr16.fits.gz',\
                        columns=['OBJID','CMODELMAG','MODELMAG','PSFMAG','FIBER2MAG'])
        photo = Table(photo_tmp)
        photo_tmp = []
        tac = join(tac_tmp,photo,keys=['OBJID'],join_type='left')

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

def get_delta_velocities_from_repeats(spall,proj,target,zmin,zmax,spec1d=0, redrock=0, spec1ddr16=0,GC='NGC+SGC'):
    filename = '{}/{}-{}_deltav_z{}z{}-{}.fits.gz'.format(home,proj,target,zmin,zmax,GC)
    if GC == 'NGC':
        spallsel = (spall['RA_REDROCK']>120)&(spall['RA_REDROCK']<240)
        spall = spall[spallsel]
    elif GC=='SGC':
        spallsel = (spall['RA_REDROCK']<120)|(spall['RA_REDROCK']>240)
        spall = spall[spallsel]

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

    # select repeats
    if os.path.exists(filename):

        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], \
                'z':[], 'zerr':[],'zerr0':[],'zerr1':[],\
                'sn_i': [], 'sn_z': [],'gi':[]}
        hdu = fits.open(filename)
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

        info = {'thids': [], 'delta_v':[], 'delta_chi2':[], \
                'z':[], 'zerr':[],'zerr0':[],'zerr1':[],\
                'sn_i': [], 'sn_z': [],\
                'cmodelmag':[],'modelmag':[],'psfmag':[],'fiber2mag':[], 'gi':[]}

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
            info['thids'].append(thid) 
            info['delta_v'].append(dv)
            info['delta_chi2'].append(dc_min)
            info['z'].append(z_clustering)
            info['sn_i'].append(sn_i)
            info['sn_z'].append(sn_z)
            # save sqrt(sum(zerr)), zerr[flag], zerr[no flag]
            info['zerr'].append(np.sqrt(spall[zerr_field][j1]**2+spall[zerr_field][j2]**2)*c_kms/(1+z_clustering))
            info['zerr1'].append((spall["SPECPRIMARY"][j2]*spall[zerr_field][j2]+spall["SPECPRIMARY"][j1]*spall[zerr_field][j1])*c_kms/(1+z_clustering))
            flaginv2 = 1-spall["SPECPRIMARY"][j2];flaginv1 = 1-spall["SPECPRIMARY"][j1]
            info['zerr0'].append((flaginv2*spall[zerr_field][j2]+flaginv1*spall[zerr_field][j1])*c_kms/(1+z_clustering))
            
            # photometric info:
            magnitudes = spall[j1]['CMODELMAG']*spall["SPECPRIMARY"][j1]+spall[j2]['CMODELMAG']*spall["SPECPRIMARY"][j2]
            info['cmodelmag'].append(magnitudes)
            info['modelmag'].append(spall[j1]['MODELMAG']*spall["SPECPRIMARY"][j1]+spall[j2]['MODELMAG']*spall["SPECPRIMARY"][j2])
            info['psfmag'].append(spall[j1]['PSFMAG']*spall["SPECPRIMARY"][j1]+spall[j2]['PSFMAG']*spall["SPECPRIMARY"][j2])
            info['fiber2mag'].append(spall[j1]['FIBER2MAG']*spall["SPECPRIMARY"][j1]+spall[j2]['FIBER2MAG']*spall["SPECPRIMARY"][j2])
            info['gi'].append(magnitudes[1]-magnitudes[3])


        # print information in this sample
        zflag = (np.array(info['z'])>zmin)&(np.array(info['z'])<zmax)
        print('{}<z<{} has {} duplicates'.format(zmin,zmax,len(np.array(info['z'])[zflag])))
        
        cols = []
        print('before:',np.array(info['z']).shape)
        for k in info.keys():
            info[k] = np.array(info[k])[zflag]
            if k.find('mag')==-1:
                cols.append(fits.Column(name=k,format='D',array=info[k]))
            else:
                cols.append(fits.Column(name=k,format='5E',array=info[k]))                
        hdulist = fits.BinTableHDU.from_columns(cols)
        hdulist.writeto(filename,overwrite=True)
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
    
def plot_deltav_hist(info,target,zrange,max_dv=500., min_deltachi2=9, nsubvolume = 1000,title=None, save=0,coloursel=False):
    #-- select inside redshift range, reject outliers
    dc = info['delta_chi2']
    dv = info['delta_v']
    z = info['z']
    zerr = info['zerr']
    w = (dc > min_deltachi2)&(abs(dv)<1000)
    dvsel = dv[w]
    
    # binning dv[w]
    binwidth = 5
    bins = np.arange(-max_dv, max_dv+1, binwidth)
    outliern = len(dv[w&(dv<-max_dv)])
    outlierp = len(dv[w&(dv>max_dv)])
    dens,BINS = np.histogram(dvsel,bins=bins)
    norm = np.sum(dens)
    dens = dens/norm
    #import pdb;pdb.set_trace()

    # directly calculate the std
    STD = jacknife_hist(dvsel,bins,nsub = nsubvolume,gaussian=False)
    print('std calculation: Vsmear = [{:.1f},{:.1f}]'.format(np.std(dvsel)-STD*np.sqrt(nsubvolume),np.std(dvsel)+STD*np.sqrt(nsubvolume)))

    ## red/blue comparison
    if coloursel:
        plt.figure(figsize=(8,6))
        bins = np.arange(-max_dv, max_dv+1, binwidth*2)
        BIN = (bins[1:]+bins[:-1])/2
        if target != 'LOWZ':
            C = info['gi']>=2.2
        else:
            C = info['gi']>=info['z']*2.8+1.2
            
        #import pdb;pdb.set_trace()
        STD = jacknife_hist(dv[w&C],bins,nsub = 100,gaussian=False)
        print('{} red galaxy std calculation: Vsmear = [{:.1f},{:.1f}]'.format(len(dv[w&C]),np.std(dv[w&C])-STD*np.sqrt(nsubvolume),np.std(dv[w&C])+STD*np.sqrt(nsubvolume)))
        dens,BINS = np.histogram(dv[w&C],bins=bins)
        norm = np.sum(dens)
        dens = dens/norm
        plt.plot(BIN,dens,'r--',label='{:.1f}% red galaxy'.format(100*len(dv[w&C])/len(dv[w])))
        
        STD = jacknife_hist(dv[w&(~C)],bins,nsub = 100,gaussian=False)
        print('{} blue galaxy std calculation: Vsmear = [{:.1f},{:.1f}]'.format(len(dv[w&(~C)]),np.std(dv[w&(~C)])-STD*np.sqrt(nsubvolume),np.std(dv[w&(~C)])+STD*np.sqrt(nsubvolume)))
        dens,BINS = np.histogram(dv[w&(~C)],bins=bins)
        norm = np.sum(dens)
        dens = dens/norm
        plt.plot(BIN,dens,'b--',label='{:.1f}% blue galaxy'.format(100*len(dv[w&(~C)])/len(dv[w])))
        plt.legend(loc=1)
        plt.title('{} {} red vs blue for {} galaxies'.format(target,zrange,len(dv[w])))
        plt.savefig(save[:-4]+'_red-blue.png', bbox_inches='tight')
        plt.close()    
    else:
        #-- fit delta_v with Gaussian, Lorentzian and Voigt line shape    
        # histogram jacknife
        BIN,hists = jacknife_hist(dvsel,bins,nsub = nsubvolume,save = save[:cutind]+target+'-'+zrange+'-maxdv'+str(max_dv)+'-jacknife-{}.dat'.format(GC))
        hists =hists/norm
        histstd = np.std(hists,axis=1)*np.sqrt(nsubvolume)
        histcovR = np.linalg.inv(np.cov(hists)*nsubvolume)*(hists.shape[1]-hists.shape[0]-2)/(hists.shape[1]-1)
        # Gaussian
        if zrange =='z0.43z0.7':
            popt, pcov = curve_fit(gaussian,BIN,dens)#,sigma=histcovR)
        else:
            popt, pcov = curve_fit(gaussian,BIN,dens,sigma=histcovR)
        res = gaussian(BIN,*popt)-dens
        print('Gaussian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,popt[1]-np.sqrt(np.diag(pcov))[1],popt[1]+np.sqrt(np.diag(pcov))[1]))      

        # fittins other than Gaussian
        """
        import pdb;pdb.set_trace()
        popt1, pcov1 = curve_fit(lorentzian,BIN,dens,sigma=histcovR)
        res1 = lorentzian(BIN,*popt1)-dens
        print('Lorentzian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,(popt1[1]-np.sqrt(np.diag(pcov1))[1])/2/np.sqrt(2*np.log(2)),(popt1[1]+np.sqrt(np.diag(pcov1))[1])/2/np.sqrt(2*np.log(2))))  

        mod = PseudoVoigtModel()
        pars = mod.guess(dens, x=BIN)
        out = mod.fit(dens, pars, x=BIN,weights=1/histstd**2)
        res2 = out.best_fit - dens
        print('Voigt fit in [-{},{}]: Vsmear = {:.1f}, {:.1f}% Lorentzian '.format(max_dv,max_dv,out.best_values['sigma'],out.best_values['fraction']))
        """

        # plot the gaussian: delta_v
        plt.figure(figsize=(8,6))
        plt.errorbar(BIN,dens,histstd,color='k', marker='o',ecolor='k',ls="none")
        plt.scatter(-max_dv,outliern/norm,c='r')
        plt.scatter(max_dv,outlierp/norm,c='r')
        plt.plot(BIN, gaussian(BIN,*popt), c='orange',label=r'Gaussian fit $\sigma = {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$/dof = {3:.1f}/{4:}'.format(popt[1],np.sqrt(np.diag(pcov))[1],np.sqrt(np.diag(pcov))[1],res.dot(histcovR.dot(res)),len(res)))
        #plt.plot(BIN, lorentzian(BIN,*popt1), label='Lorentzian fit '+r'$\frac{w}{2\sqrt{2ln2}}$'+'$= {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$,$\chi^2$ /dof = {3:.1f}/{4:}'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2)),res1.dot(histcovR.dot(res1)),len(res1)))
        #plt.plot(BIN, out.best_fit,c='green',label=r'PseudoVoigt fit $\sigma$ = {0:.1f}, {1:.1f}% Lorentzian, $\chi^2$/dof = {2:.1f}/{3:}'.format(out.best_values['sigma'],out.best_values['fraction'],res2.dot(histcovR.dot(res2)),len(res)))
        plt.xlabel(r'$\Delta v$ (km/s)')
        plt.ylabel('counts')
        #plt.yscale('log')
        plt.ylim(0,max(dens)*1.3)
        plt.legend(loc=1)    
        if title:
            plt.title(title+' with {} pairs, std = {:.1f} $\pm$ {:.1f}, fitting by curve_fit'.format(dvsel.size,np.std(dv[abs(dv)<1000]),STD*np.sqrt(nsubvolume)))
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight')

        plt.close()
        
        #-- fit a Gaussian for zerr/Delta v
        ratios = np.linspace(-4,4,81)
        ratio = (ratios[1:]+ratios[:-1])/2
        ratiodens,ratiobin = np.histogram(dv/zerr,ratios)
        ratiodens = ratiodens/len(ratiodens)
        
        BIN,hists = jacknife_hist(dv/zerr,ratios,nsub = nsubvolume,save = save[:cutind]+target+'-'+zrange+'-maxdv'+str(max_dv)+'-jacknife-zerr-{}.dat'.format(GC))
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
        plt.tight_layout()
        if save:
            plt.savefig(save[:-4]+'-zerr-{}.png'.format(GC), bbox_inches='tight')
        plt.close()

def plot_all_deltav_histograms(spall,proj,zmin,zmax,target='LRG',dchi2=9,maxdv=500,spec1d=0, redrock=0,GC='NGC+SGC',coloursel=False):

    spall = Table.read(spall)
    sp = get_targets(spall, target=target)
    import pdb;pdb.set_trace()
    if spec1d:
        zsource = 'spec1d'
        info = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,spec1d=1,GC=GC)
    elif redrock:
        zsource = 'redrock'
        info = get_delta_velocities_from_repeats(sp,proj,target,zmin,zmax,redrock=1,GC=GC)
        
    plot_deltav_hist(info,target,zrange='z{}z{}'.format(zmin,zmax),min_deltachi2=dchi2,  max_dv=maxdv,title='{} {} {}<z<{}'.format(proj,target,zmin,zmax), save='{}/{}-{}-repeats-{}-dchi2_{}-z{}z{}-{}.png'.format(home,proj,target,zsource,dchi2,zmin,zmax,GC),coloursel=coloursel)
    #plot_deltav_hist_emcee(info,target,zrange='z{}z{}'.format(zmin,zmax),min_deltachi2=dchi2,  max_dv=maxdv,title='{} {} {}<z<{}'.format(proj,target,zmin,zmax), save='{}/{}-{}-repeats-{}-dchi2_{}-z{}z{}-{}.png'.format(home,proj,target,zsource,dchi2,zmin,zmax,GC))
        
##=================================================================================

#write_spall_redrock_join('spAll-v5_13_0.fits', 'spAll_trimmed_pREDROCK.fits','spAll-zbest-v5_13_0-photo.fits') # with photo info
#write_spall_repeats('spAll-zbest-v5_13_0-photo.fits', 'spAll-zbest-v5_13_0-repeats-2x_redrock-photo.fits') # with photo-info
#write_spall_repeats('specObj-dr16.fits', 'spAll-zbest-dr16-repeats-2x_LOWZ.fits') # two populations of LOWZ: before MJD (zerr=0) and after
#write_spall_repeats('spAll-v5_4_45.fits', 'spAll-zbest-v5_4_45-repeats-2x.fits') # an older version
#plot_all_deltav_histograms('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','BOSS',zmin=0.2,zmax=0.43,target='LOWZ',dchi2=9,spec1d=1,maxdv=140)

GC = 'NGC+SGC'
repeatname = 'spAll-zbest-v5_13_0-repeats-2x_redrock-photo.fits'

zmins = [0.6,0.6,0.65,0.7,0.8,0.6]
zmaxs = [0.7,0.8,0.8, 0.9,1.0,1.0]
maxdvs = [235,275,275,300,255,360]
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    plot_all_deltav_histograms(repeatname,'eBOSS',zmin,zmax,target='LRG',dchi2=9,redrock=1,maxdv=maxdv,coloursel=True)

zmins = [0.43,0.51,0.57,0.43]
zmaxs = [0.51,0.57,0.7,0.7]
maxdvs = [205,200,235,270]
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    plot_all_deltav_histograms(repeatname,'BOSS',zmin,zmax,target='CMASS',dchi2=9,spec1d=1,maxdv=maxdv,coloursel=True)

    
zmins = [0.2, 0.33,0.2]
zmaxs = [0.33,0.43,0.43]
if GC == 'NGC':
    maxdvs = [85,120,130] #NGC
elif GC == 'SGC':
    maxdvs = [105,135,135] #SGC
else:
    maxdvs = [105,140,140] # NGC+SGC
for zmin,zmax,maxdv in zip(zmins,zmaxs,maxdvs):
    plot_all_deltav_histograms(repeatname,'BOSS',zmin,zmax,target='LOWZ',dchi2=9,spec1d=1,maxdv=maxdv,GC=GC,coloursel=True)
    #plot_all_deltav_histograms('spAll-zbest-dr16-repeats-2x_LOWZ.fits','BOSS',zmin,zmax,target='LOWZdr16',dchi2=9,spec1d=1,maxdv=maxdv)

###############################################################################################   
"""
## dv[sel] related quantities are not corrected
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
    BIN,hists = jacknife_hist(dvsel,bins,nsub = nsubvolume,save = save[:cutind]+target+'-'+zrange+'-maxdv'+str(max_dv)+'-jacknife-{}.dat'.format(GC))
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
    print('Gaussian emcee fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,sigma[0]-sigma[1],sigma[0]+sigma[2]))  
    # plot the posterior
    import corner
    fig = corner.corner(samples, labels=[r'a', r'$\sigma$', r'$\mu$'])
    plt.savefig(save[:-4]+'_posteriorG-{}.png'.format(GC))
    plt.close()
    
    #######################################################################
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
    
    print('Lorentzian fit in [-{},{}]: Vsmear = [{:.1f},{:.1f}]'.format(max_dv,max_dv,(w[0]-w[1])/2/np.sqrt(2*np.log(2)),(w[0]+w[2])/2/np.sqrt(2*np.log(2))))
    # plot the posterior
    fig = corner.corner(samples1, labels=[r'a', r'w', r'$p_0$'])
    plt.savefig(save[:-4]+'_posteriorL.-{}.png'.format(GC))
    plt.close()
    #######################################################################
    
    # plot the Delta v vs models
    plt.figure(figsize=(8,6))
    plt.errorbar(BIN,dens,histstd,color='k', marker='o',ecolor='k',ls="none")
    plt.scatter(-max_dv,outliern/norm,c='r')
    plt.scatter(max_dv,outlierp/norm,c='r')
    plt.plot(BIN, gaussian(BIN,a[0],sigma[0],mu[0]), label=r'Gaussian fit $\sigma = {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(sigma[0],sigma[1],sigma[2],res.dot(histcovR.dot(res)),len(res)))
    #plt.plot(BIN, lorentzian(BIN,a1[0],w[0],p[0]),label='Lorentzian fit '+r'$\frac{w}{2\sqrt{2ln2}}$'+'$= {0:.1f}_{{-{1:.2f}}}^{{+{2:.2f}}}$, $\chi^2$ /dof = {3:.1f}/{4:}'.format(w[0]/2/np.sqrt(2*np.log(2)),w[1]/2/np.sqrt(2*np.log(2)),w[2]/2/np.sqrt(2*np.log(2)),res1.dot(histcovR.dot(res1)),len(res1)))
    plt.xlabel(r'$\Delta v$ (km/s)')
    plt.ylabel('normalised counts')
    plt.legend(loc=1)
    #plt.yscale('log')
    plt.ylim(0,max(dens)*1.3)
    if title:
        plt.title(title+' with {} pair, fitting by emcee'.format(dvsel.size))
    plt.tight_layout()
    if save:
        plt.savefig(save[:-4]+'_emcee-{}.png'.format(GC), bbox_inches='tight')
    plt.close()
"""
##############################################################################################
"""
#plot_all_deltav_deltachi2('spAll-zbest-v5_13_0-repeats-2x_redrock.fits','eBOSS',zmin,zmax,target='LRG',dchi2=9,redrock=1)
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
               save='{}/{}-{}-repeats-{}-dchi2_{}-z{}z{}.pdf'.format(home,proj,target,zsource,dchi2,zmin,zmax))
"""
#########################################################################################    
