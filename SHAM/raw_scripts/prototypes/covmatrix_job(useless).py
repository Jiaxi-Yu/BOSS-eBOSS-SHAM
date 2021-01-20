# Produce bins file, task list and submit the task 
#for both linear and logarithmic binning and NGC, SGC
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import os
import glob

# Variables(should be the argument of the function)
#home='/home/epfl/jiayu/Desktop/master/'
#path='/home/epfl/zhaoc/data/EZmock/syst/EZmock_eBOSS_LRGpCMASS_v7.0_syst/'
#mockfile[f][-44:-10]
home = '/global/cscratch1/sd/jiaxi/master/'
path = '/global/homes/z/zhaoc/cscratch/EZmock/LRG/EZmock_eBOSS_LRG_v7.0_syst/'
rmin=0.1
rmax=200
nbins=40


# remove bin files to avoid the overwriting problem
code = home+'FCFC/'
if os.path.exists(code+'bins_linear.dat'):
	os.remove(code+'bins_linear.dat')

if os.path.exists(code+'bins_log.dat'):
	os.remove(code+'bins_log.dat')

# create bins files
rbins = np.linspace(rmin, rmax, nbins + 1)
for i in range(nbins):
        file=open(code+'bins_linear.dat',mode='a+')
        file.write(str(rbins[i])+'\t'+str(rbins[i+1])+'\n')
        file.close()

rbins1 = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
for i in range(nbins):
        file=open(code+'bins_log.dat',mode='a+')
        file.write(str(rbins1[i])+'\t'+str(rbins1[i+1])+'\n')
        file.close()


# create task list
for bins in ['linear','log']:
	binfile  =('bins_'+bins+'.dat')
	for GC in ['NGC','SGC']:
		output ='2PCF/chi2/'+bins+'/'+GC+'_'
		tasklist = code+'task_'+GC+'_'+bins+'.txt'
		mockfile = glob.glob(path+'*'+GC+'*.dat*')
		# remove existing task list
		if os.path.exists(tasklist):
			os.remove(tasklist)
		for f in range(len(mockfile)):
		#for f in range(2):             
			name = open(tasklist,mode='a+')
			name.write('./2pcf -c fcfc.conf -q '+binfile+' --data '+mockfile[f]+' --rand '+mockfile[f][:-10]+'.ran.ascii --dd ../'+output+mockfile[f][-14:-10]+'.dd --dr ../'+output+mockfile[f][-14:-10]+'.dr --rr ../'+output+mockfile[f][-14:-10]+'.rr --output ../'+output+mockfile[f][-14:-10]+'.dat'+'\n')
			name.close()


os.system('sbatch cftask.sh')

./OMPrun task_NGC_linear.txt
./OMPrun task_NGC_log.txt
./OMPrun task_SGC_linear.txt
./OMPrun task_SGC_log.txt
