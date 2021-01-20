import matplotlib 
matplotlib.use('agg')
from getdist import plots, MCSamples, loadMCSamples
import getdist
import matplotlib.pyplot as plt
import pymultinest

# variables
gal      = 'ELG'
GC       = 'NGC'
date     = '0526' 
npoints  = 200#int(sys.argv[3])
nseed    = 15
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
home      = '/global/cscratch1/sd/jiaxi/master/'
fileroot1 = 'MCMCout/'+date+'/HAM_ELG_NGC/multinest_'
fileroot2 = 'MCMCout/'+date+'/HAM_ELG_SGC/multinest_'

parameters = ["sigma","vcut"]
npar = len(parameters)
a1 = pymultinest.Analyzer(npar, outputfiles_basename = fileroot1)
a2 = pymultinest.Analyzer(npar, outputfiles_basename = fileroot2)

sample1 = loadMCSamples(fileroot1)
sample2 = loadMCSamples(fileroot2)
plt.rcParams['text.usetex'] = False
g = plots.get_single_plotter()
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add=0.4
#g.settings.title_limit_fontsize = 14
g = plots.get_subplot_plotter()
g.triangle_plot([sample1,sample2],['sigma', 'vcut'], filled=False,legend_labels=['NGC', 'SGC'])
g.subplots[0,0].axvline(a1.get_best_fit()['parameters'][0], color='gray', ls='--')
g.subplots[0,0].axvline(a2.get_best_fit()['parameters'][0], color='red', ls='--')
g.subplots[1,1].axvline(a1.get_best_fit()['parameters'][1], color='grey', ls='--')
g.subplots[1,1].axvline(a2.get_best_fit()['parameters'][1], color='red', ls='--')

g.export('posterior_ELG.png')
plt.close('all')

