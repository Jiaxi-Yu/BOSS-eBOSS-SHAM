from astropy.table import Table
import numpy as np
from glob import glob
import sys

gal= sys.argv[1]
if gal=='LRG':
    route = '/global/homes/z/zhaoc/cscratch/EZmock/LRG/EZmock_eBOSS_LRG_v7/EZmock_eBOSS_LRG_{}_v7_{}.dat'
    rand = '/global/homes/z/zhaoc/cscratch/EZmock/LRG/EZmock_eBOSS_LRG_v7/random_20x_eBOSS_LRG_{}_v7.dat'
    ver = 'v7_2'

else:
    route = '/global/homes/z/zhaoc/cscratch/EZmock/ELG/nosel_v5/fkp_cap/EZmock_eBOSS_ELG_{}_v7_{}.dat'
    rand = '/global/homes/z/zhaoc/cscratch/EZmock/ELG/nosel_v5/fkp_cap/random_20x_eBOSS_ELG_{}_v7.dat'
    ver='v7'

home='/global/cscratch1/sd/jiaxi/master/catalog/nersc_wp_{}_{}/EZmocks/'.format(gal,ver)
files = glob(route.format('NGC','*'))
Ngal = np.zeros((len(files),2))
for i,GC in enumerate(['NGC','SGC']):
    for N in range(len(files)):
        f = open('{}param_{}_{}_{}.ini'.format(home,gal,GC,str(N+1).zfill(4)),'w')
        f.write('# input-output files and parameters\n')
        f.write('data_filename= {}\n'.format(route.format(GC,str(N+1).zfill(4))))
        f.write('random_filename= {}\n'.format(rand.format(GC)))
        f.write('#The next 2 files are only needed if you want to\n')
        f.write('#cross-correlate two datasets\n')
        f.write('data_filename_2= {}\n'.format(route.format(GC,str(N+1).zfill(4))))
        f.write('random_filename_2= {}\n'.format(rand.format(GC)))
        f.write('input_format= 2\n')
        f.write('output_filename= {}/EZmock_{}_{}_{}.dat\n'.format(home,gal,GC,str(N+1).zfill(4)))
        f.write('\n')
        f.write('# estimation parameters\n')
        f.write('corr_type= 3D_ps\n')
        f.write('\n')
        f.write('# cosmological parameters\n')
        f.write('omega_M= 0.31\n')
        f.write('omega_L= 0.69\n')
        f.write('w= -1\n')
        f.write('\n')
        f.write('# binning\n')
        f.write('log_bin= 1\n')
        f.write('dim1_max= 44.66835921509633\n')
        f.write('dim1_min_logbin= 0.0891251\n')
        f.write('dim1_nbin= 27\n')
        f.write('dim2_max= 80.\n')
        f.write('dim2_nbin= 80\n')
        f.write('dim3_min= 0.4\n')
        f.write('dim3_max= 0.7\n')
        f.write('dim3_nbin= 1\n')
        f.write('\n')
        f.write('# pixels for radial correlation\n')
        f.write('radial_aperture= 1.\n')
        f.write('\n')
        f.write('# pm parameters\n')
        f.write('use_pm= 0\n')
        f.write('n_pix_sph= 2048\n')
        f.close()
