import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

import evalSR
import importlib
importlib.reload(evalSR)

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dz,z,zrsel,ggzsel = {'LRG':[],'ELG':[],'QSOlz':[],'BGS':[]},{'LRG':[],'ELG':[],'QSOlz':[],'BGS':[]},{'LRG':[],'ELG':[],'QSO':[],'BGSlz':[]},{'LRG':[],'ELG':[],'QSOlz':[],'BGS':[]}
#-- fit a Gaussian
def gaussian(x,a,sigma,mu):
    return a/np.sqrt(2*np.pi)/sigma*np.exp(-(x-mu)**2/(2*sigma**2))
def lorentzian(x,a,w,p):
    return a/(1+((x-p)*2/w)**2)

TYPE = 'LRG'
if not os.path.exists('{}_deltav.dat'.format(TYPE)):
    tf = evalSR.add_truth(TYPE,release='cascades',version='3',bdir='/global/cfs/cdirs/desi/survey/catalogs/SV1/redshift_comps')
    # zrsel for the redshift-range selection, ggsel for the redshift-range and removing-catastrophic selection
    dz[TYPE],z[TYPE],zrsel[TYPE],ggzsel[TYPE] = evalSR.repeatvsdchi2(tf,TYPE)
    dv = dz[TYPE][ggzsel[TYPE]]
    np.savetxt('{}_deltav.dat'.format(TYPE),dv)
else:
    dv = np.loadtxt('{}_deltav.dat'.format(TYPE))


maxdv = 400
bins = np.linspace(-maxdv,maxdv,maxdv*2+1)
bins = bins
fig,ax    = plt.subplots(1,1,figsize=(8,6))
dens,BINS,plot = plt.hist(dv, bins=bins, histtype='step',alpha=0.6)
BIN = (bins[1:]+bins[:-1])/2
popt, pcov = curve_fit(gaussian,BIN,dens)    
popt1, pcov1 = curve_fit(lorentzian,BIN,dens)

from lmfit.models import PseudoVoigtModel
mod = PseudoVoigtModel()
pars = mod.guess(dens, x=BIN)
pars['fraction'].set(value=0.1, min=0, max=0.4)
out = mod.fit(dens, pars, x=BIN)
print(out.fit_report(min_correl=0.25))

plt.plot(BIN, gaussian(BIN,*popt),c='r',label=r'Gaussian $\mu = {:.1f} \pm {:.1f}, \sigma = {:.1f} \pm {:.1f}$'.format(popt[2],np.sqrt(np.diag(pcov))[2], popt[1],np.sqrt(np.diag(pcov))[1]))
plt.plot(BIN, lorentzian(BIN,*popt1),c='k',label='Lorentzian $p_0 = {:.1f} \pm {:.1f}$, '.format(popt1[2],np.sqrt(np.diag(pcov1))[2])+r'w/(2$\sqrt{2ln2})$'+' = ${:.1f} \pm {:.1f}$'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2))))
plt.plot(BIN, out.best_fit,c='green',label='PseudoVoigt')# $p_0 = {:.1f} \pm {:.1f}$, '.format(popt1[2],np.sqrt(np.diag(pcov1))[2])+r'w/(2$\sqrt{2ln2})$'+' = ${:.1f} \pm {:.1f}$'.format(popt1[1]/2/np.sqrt(2*np.log(2)),np.sqrt(np.diag(pcov1))[1]/2/np.sqrt(2*np.log(2))))
outliern = len(dv[(dv>-1000)&(dv<-maxdv)])
outlierp = len(dv[(dv<1000)&(dv>maxdv)])
plt.scatter(-maxdv,outliern,c='b',label='outliers')
plt.scatter(maxdv,outlierp,c='b')
plt.title(TYPE+' dv histogram ,stdev = {:.1f} km/s'.format(np.std(dv)))
ax.set_xlim(-maxdv-5,maxdv+5)
plt.ylim(1e-1,10*max(dens))
plt.yscale('log')
ax.set_xlabel('$\Delta$ v (km/s)')
ax.set_ylabel('counts')
ax.grid(True)
plt.legend(loc=0)
plt.savefig('{}_deltav_hist_std{:.1f}.png'.format(TYPE,np.std(dv)))
plt.close()
print('std: {:.1f}'.format(np.std(dv)))