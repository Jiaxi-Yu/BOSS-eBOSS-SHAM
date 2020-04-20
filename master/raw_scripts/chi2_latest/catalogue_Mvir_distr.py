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
MultiDark = fits.open(path+'MD_hlist_0.53780.fits.gz')
UNIT      = fits.open(path+'UNIT_hlist_0.53780.fits.gz')
# halo(PID==-1) or subhalo(PID!=-1)
MDP = MultiDark[1].data['PID']
UNIP =UNIT[1].data['PID']
#function = sys.argv[1] # 'halo'  'subhalo' 'mix'
# Mvir and Vpeak

for i,function in enumerate(['mix','halo','subhalo']):
    if function == 'halo':
        MD = MultiDark[1].data['Mvir'][MDP==-1]
        UNI =UNIT[1].data['Mvir'][UNIP==-1]
        MDV = MultiDark[1].data['Vpeak'][MDP==-1]
        UNIV =UNIT[1].data['Vpeak'][UNIP==-1]
    elif function =='subhalo':
        MD = MultiDark[1].data['Mvir'][MDP!=-1]
        UNI =UNIT[1].data['Mvir'][UNIP!=-1]
        MDV = MultiDark[1].data['Vpeak'][MDP!=-1]
        UNIV =UNIT[1].data['Vpeak'][UNIP!=-1]
    else:
        MD = MultiDark[1].data['Mvir']
        UNI =UNIT[1].data['Mvir']
        MDV = MultiDark[1].data['Vpeak']
        UNIV =UNIT[1].data['Vpeak']

    # Mvir distribution
    Mbins=np.logspace(np.log10(10**9.3),np.log10(10**15.1),50+1) 
    num1 = np.histogram(UNI,bins=Mbins)#,alpha=0.5,label='UNIT',color='r')
    num2 = np.histogram(MD,bins=Mbins)#,alpha=0.5,label='MultiDark',color='b')
    # store the data
    file = 'Mvir_distribution_'+function+'.txt'
    if os.path.exists(file)==False: 
        ascii.write(Table([Mbins[:-1],Mbins[1:],num1[0],num2[0]],names=('M_min', 'M_max', 'MD', 'UNIT')),file,delimiter='\t',overwrite=True)


    # vpeak distribution function
    Vbins=np.linspace(20,1500,50+1) 
    num11 = np.histogram(UNIV,bins=Vbins)#,alpha=0.5,label='UNIT',color='r')
    num21 = np.histogram(MDV,bins=Vbins)#,alpha=0.5,label='MultiDark',color='b')
    # store the data
    file2 = 'vmax_distribution_'+function+'.txt'
    if os.path.exists(file2)==False:
        ascii.write(Table([Vbins[:-1],Vbins[1:],num11[0],num21[0]],names=('V_min', 'V_max', 'MD', 'UNIT')),file2,delimiter='\t',overwrite=True)



    # Mvir distribution plot
    fig = plt.figure(figsize = (7, 8)) 
    spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[4, 1,1], hspace=0.3)
    ax = np.empty((3,1), dtype=type(plt.axes))
    for k,value in enumerate([np.ones(50),num2[0],num2[0]]):
        ax[k,0] = fig.add_subplot(spec[k,0])

        ax[k,0].step(Mbins[:-1],num1[0]/value,c='r',label = 'MultiDark',alpha=0.5)
        ax[k,0].step(Mbins[:-1],num2[0]/value,c='b',label = 'UNIT',alpha=0.5)
            
        ax[k,0].set_xlabel('$M_{vir}(M_{sun}/h)$')
        ax[k,0].set_xlim(Mbins[0],Mbins[-1])
        if k==0:
            plt.title('the '+function+' Mvir distribution comparison') 
            ## title cannot be set for the whole figure(outside of the loop) 
            ## otherwise, subplots will stick together.
            ax[k,0].set_ylim(1e0,1e8)
            ax[k,0].set_ylabel('# of haloes') 
            plt.xscale('log');
            plt.yscale('log')
            plt.legend(loc=0) 
        elif k==1:
            ax[k,0].set_ylim(0.5,1.5)
            ax[k,0].set_ylabel('halo # ratio')
            plt.xscale('log');
        else:
            ax[k,0].set_ylim(0,4)
            ax[k,0].set_yticks([0,1,2,3,4])
            ax[k,0].set_ylabel('halo # ratio')
            plt.xscale('log');
                
    plt.savefig(path[:-8]+'Mvir_distribution_'+function+'.png')
    plt.close()

    # vmax distribution plot
    fig = plt.figure(figsize = (7, 8)) 
    spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[4,1, 1], hspace=0.3)
    ax = np.empty((3,1), dtype=type(plt.axes))
    for k,value in enumerate([np.ones(50),num21[0],num21[0]]):
        ax[k,0] = fig.add_subplot(spec[k,0])

        ax[k,0].step(Vbins[:-1],num11[0]/value,c='r',label = 'MultiDark',alpha=0.5)
        ax[k,0].step(Vbins[:-1],num21[0]/value,c='b',label = 'UNIT',alpha=0.5)
            
        ax[k,0].set_xlabel('$V_{max}(km/s)$')
        ax[k,0].set_xlim(Vbins[0],Vbins[-1])
        if k==0:
            plt.title('the '+function+' Vmax distribution comparison')
            ax[k,0].set_ylim(1e0,3e8)
            ax[k,0].set_ylabel('# of haloes') 
            plt.yscale('log')
            plt.legend(loc=0) 
        elif k==1:
            ax[k,0].set_ylim(0.5,1.5)
            ax[k,0].set_ylabel('halo # ratio')
        else:
            ax[k,0].set_ylim(0,4)
            ax[k,0].set_yticks([0,1,2,3,4])
            ax[k,0].set_ylabel('halo # ratio')
                
    plt.savefig(path[:-8]+'Vmax_distribution_'+function+'.png')
    plt.close()
time_end=time.time()
print('plotting the Mvir and Vmax distribution costs',time_end-time_start,'s')



# select subhaloes
MDtot = MultiDark[1].data[MDP!=-1] 
UNItot =UNIT[1].data[UNIP!=-1]

# read the parent galaxy ID from heave subhaloes(>10**14)
threshold = 3e13
IDMD = MDtot['PID'][MD>threshold]
IDUNI = UNItot['PID'][UNI>threshold]

# find corresponding index of parent haloes
## np.in1d(A,B) won't select haloes that have multiple subhaloes
## np.searchsorted(A,B) can do the above task, but A should be ascending.
#MDind = np.where(np.in1d(MultiDark[1].data['ID'], IDMD))[0]
ascMD = np.argsort(MultiDark[1].data['ID'])
ascUN = np.argsort(UNIT[1].data['ID'])
MDind = np.searchsorted(MultiDark[1].data['ID'][ascMD], IDMD)
UNIind = np.searchsorted(UNIT[1].data['ID'][ascUN], IDUNI)

# a small tests about the correctness of selecting:
np.all(IDMD==MultiDark[1].data['ID'][ascMD][MDind])

# conclude the results
MDparent, UNIparent = np.zeros((len(IDMD),2)),np.zeros((len(IDUNI),2))
MDparent[:,0] = MDtot['Mvir'][MDtot['Mvir']>threshold]
MDparent[:,1] = MultiDark[1].data['Mvir'][ascMD][MDind]

UNIparent[:,0] = UNItot['Mvir'][UNItot['Mvir']>threshold]
UNIparent[:,1] = UNIT[1].data['Mvir'][ascUN][UNIind]

fig,ax = plt.subplots()
plt.scatter(MDparent[:,0],MDparent[:,1],c='r',alpha=0.4,s=5,label= ('MultiDark('+str(len(IDMD))+' subhaloes)'))
plt.scatter(UNIparent[:,0],UNIparent[:,1],c='b',alpha=0.4,s=5,label=('UNIT('+str(len(IDUNI))+' subhaloes)'))
diff1 = (MDparent[:,1]-MDparent[:,0])>0
diff2 = (UNIparent[:,1]-UNIparent[:,0])>0
plt.scatter(MDparent[:,0][~diff1],MDparent[:,1][~diff1],c='r',alpha=0.5,s=20)
plt.scatter(UNIparent[:,0][~diff2],UNIparent[:,1][~diff2],c='b',alpha=0.5,s=20)
plt.xlabel('subhalo mass ($M_{sun}/h$)')
plt.ylabel('parent halo mass ($M_{sun}/h$)')
plt.xscale('log')
plt.yscale('log')
plt.xlim(2.8e13,5*10**14)
plt.ylim(2.5e13,1.2*10**15)
plt.plot(np.logspace(13,14.7,6),np.logspace(13,14.7,6),'k--',label='$M_{sub}=M_{parent}$')
plt.legend(loc=4)
plt.title('subhaloes - parent haloes relations  ($M_{vir,sub}<3e13 M_{sun}/h$)')
plt.savefig('subhalo-parent.png')



# the vpeak-mvir relation(inappropriate)
'''
mean,median = np.zeros((50,2)),np.zeros((50,2)) 
for i in range(50): 
    median[i,0]=np.nanmedian(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    median[i,1]=np.nanmedian(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])]) 
    mean[i,0]=np.nanmean(MultiDark[1].data['Vpeak'][(MD>bins[i])&(MD<bins[i+1])]) 
    mean[i,1]=np.nanmean(UNIT[1].data['Vpeak'][(UNI>bins[i])&(UNI<bins[i+1])])
# plot the vpeak-mvir relation(inappropriate)
fig = plt.figure(figsize = (7, 6)) 
spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[4, 1], hspace=0.3)
ax = np.empty((2,1), dtype=type(plt.axes))
for k,value in enumerate([np.ones(50),median[:,0]]):
    ax[k,0] = fig.add_subplot(spec[k,0])

    ax[k,0].plot(bins[:-1],median[:,0]/value,c='r',label = 'MD_median')
    ax[k,0].plot(bins[:-1],median[:,1]/value,c='b',label = 'UNIT_median')
        

    ax[k,0].set_xlabel('$M_{vir}$')
    if k==0:
        ax[k,0].set_ylim(0,2000)
        ax[k,0].set_ylabel('stacked $V_{peak}$') 
        plt.xscale('log');
        plt.legend(loc=0) 
    else:
        ax[k,0].set_ylim(-100,100)
        ax[k,0].set_ylabel('$V_{peak}$ diff')
        plt.xscale('log');
            
plt.savefig(path[:-8]+'Vmax-Mvir_cmoparison_median_test.png')
plt.close()
'''