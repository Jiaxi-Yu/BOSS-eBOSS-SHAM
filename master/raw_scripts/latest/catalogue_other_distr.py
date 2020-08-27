import matplotlib 
matplotlib.use('agg')
import time
time_start=time.time()
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from astropy.io import fits
import matplotlib.gridspec as gridspec
import os
import sys

# data reading
path =  '/global/cscratch1/sd/jiaxi/master/catalog/'
MultiDark = fits.open(path+'MD_hlist_0.53780-subhalo.fits.gz')
UNIT      = fits.open(path+'UNIT_hlist_0.53780-subhalo.fits.gz')
# halo(PID==-1) or subhalo(PID!=-1)
MDP = MultiDark[1].data['PID']
UNIP =UNIT[1].data['PID']
#function = sys.argv[1] # 'halo'  'subhalo' 'mix'
#quantity = sys.argv[1] # 'M500c', 'M200c','Vpeak'

for quantity in ['Mvir','Vmax','M500c', 'M200c','Vpeak']:#['Vpeak','Vmax']:#
    fig = plt.figure(figsize = (21, 8)) 
    spec = gridspec.GridSpec(ncols=3, nrows=3, height_ratios=[4, 1,1], hspace=0.3,wspace=0.5)
    ax = np.empty((3,3), dtype=type(plt.axes))

    for i,function in enumerate(['mix','halo','subhalo']):
        if function == 'halo':
            MD = MultiDark[1].data[quantity][MDP==-1]
            UNI =UNIT[1].data[quantity][UNIP==-1]
        elif function =='subhalo':
            MD = MultiDark[1].data[quantity][MDP!=-1]
            UNI =UNIT[1].data[quantity][UNIP!=-1]
        else:
            MD = MultiDark[1].data[quantity]
            UNI =UNIT[1].data[quantity]
            
        # Mvir distribution
        if (quantity=='M500c')|(quantity=='M200c'):
            Mbins=np.logspace(8,np.log10(UNI.max()),50+1) 
        elif (quantity=='Vmax')|(quantity=='Vpeak'):
            Mbins=np.linspace(UNI.min(),UNI.max(),50+1) 
        else:
            Mbins=np.logspace(np.log10(UNI.min()),np.log10(UNI.max()),50+1) 
        num1 = np.histogram(UNI,bins=Mbins)#,alpha=0.5,label='UNIT',color='r')
        num2 = np.histogram(MD,bins=Mbins)#,alpha=0.5,label='MultiDark',color='b')
        # store the data
        file = quantity+'_distribution_'+function+'.txt'
        if os.path.exists(file)==False: 
            ascii.write(Table([Mbins[:-1],Mbins[1:],num1[0],num2[0]],names=('bin_min', 'bin_max', 'MD', 'UNIT')),file,delimiter='\t',overwrite=True)

        # quantity distribution plot

        for k,value in enumerate([np.ones(50),num2[0],num2[0]]):
            ax[k,i] = fig.add_subplot(spec[k,i])
            ax[k,i].step(Mbins[:-1],num1[0]/value,c='r',label = 'MultiDark',alpha=0.5)
            ax[k,i].step(Mbins[:-1],num2[0]/value,c='b',label = 'UNIT',alpha=0.5)
            if quantity[0]=='M':
                plt.xscale('log');
                ax[k,i].set_xlabel(quantity+'$(M_{sun}/h)$')
            else:
                ax[k,i].set_xlabel(quantity+'$(km/s)$')
            ax[k,i].set_xlim(Mbins[0],Mbins[-1])
            if k==0:
                plt.title('the '+function+' '+quantity+' distribution comparison') 
                ## title cannot be set for the whole figure(outside of the loop) 
                ## otherwise, subplots will stick together.
                ax[k,i].set_ylim(1e0,1e8)
                ax[k,i].set_ylabel('# of haloes')
                plt.yscale('log')
                plt.legend(loc=0) 
            elif k==1:
                ax[k,i].set_ylim(0.5,1.5)
                ax[k,i].set_ylabel('halo # ratio')
            elif k==2:
                ax[k,i].set_ylim(0,4)
                ax[k,i].set_yticks([0,1,2,3,4])
                ax[k,i].set_ylabel('halo # ratio')
                
    plt.savefig(path[:-8]+quantity+'_distribution.png')
    plt.close()

time_end=time.time()
print('plotting the Mvir and Vmax distribution costs',time_end-time_start,'s')

# Vpeak-M500 relation
MDV = MultiDark[1].data['Vpeak'][MDP!=-1]
MDM = MultiDark[1].data['M500c'][MDP!=-1]
UNV = UNIT[1].data['Vpeak'][UNIP!=-1]
UNM = UNIT[1].data['M500c'][UNIP!=-1]
fig,ax = plt.subplots()
plt.scatter(MDM,MDV,c='r',alpha=0.4,s=5,label = 'MultiDark')
plt.scatter(UNM,UNV,c='b',alpha=0.4,s=5,label = 'UNIT')
plt.xlabel('M500c ($M_{sun}/h$)')
plt.ylabel('Vpeak ($km/s$)')
plt.xscale('log')
plt.xlim(1e8,5e14)
#plt.ylim(2.5e13,1.2*10**15)
#plt.plot(np.logspace(13,14.7,6),np.logspace(13,14.7,6),'k--',label='$M_{sub}=M_{parent}$')
plt.legend(loc=0)
plt.title('Vpeak - M500c relation for subhaloes')
plt.savefig('Vpeak - M500c.png')

# plot subhalo & halo Vpeak only for UNIT comparison
quantity ='Vpeak'
color = ['b','r']
rout='E:/Master/OneDrive/master_thesis/master/halo_catalogue_comparison/halo-subhalo-mix_other_columns'
fig = plt.figure(figsize = (7, 8))
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4,1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
num2 = [1,2]
for i,function in enumerate(['subhalo','halo']):
    if function == 'halo':
        UNI =UNIT[1].data[quantity][UNIP==-1]
    elif function =='subhalo':
        UNI =UNIT[1].data[quantity][UNIP!=-1]
    Mbins=np.linspace(20,1500,50+1) 
    num2[i],a = np.histogram(UNI,bins=Mbins)

    # quantity distribution plot
    for k,value in enumerate([np.ones(50),num2[0]]):
        ax[k,0] = fig.add_subplot(spec[k,0])
        ax[k,0].step((Mbins[:-1]+Mbins[1:])/2,num2[i]/(value+1),c=color[i],label = function)
        ax[k,0].set_xlabel(quantity+'$(km/s)$')
        ax[k,0].set_xlim(Mbins[0],Mbins[-1])
        if k==0:
            plt.title('the '+function+' '+quantity+' distribution comparison') 
            ## title cannot be set for the whole figure(outside of the loop) 
            ## otherwise, subplots will stick together.
            ax[k,0].set_ylim(1e0,1e8)
            ax[k,0].set_ylabel('# of haloes')
            plt.yscale('log')
            plt.legend(loc=0) 
        elif k==1:
            ax[k,0].set_ylabel('halo # ratio')
            
            plt.yscale('log')
            
plt.savefig(quantity+'.png')
plt.close()


