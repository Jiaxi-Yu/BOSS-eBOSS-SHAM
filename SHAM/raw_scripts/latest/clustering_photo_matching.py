#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
import os
#from kcorrection import kcorr
#import kcorrect 
#kcorrect.load_templates()
#kcorrect.load_filters()

home = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
filename = home+'/{}_{}_mag.fits.gz'
def zsel(DATA,zmin,zmax):
    return (DATA['z_spec']>zmin)&(DATA['z_spec']<zmax)
def colcut(DATA):
    if gal == 'CMASS':
        colsel = DATA['gi_spec']<2.35
    elif gal == 'LOWZ':
        colsel = DATA['gi_spec']<DATA['z_spec']*2.8+1.1
    return colsel

if not os.path.exists(filename.format('LOWZ','South')):
    # photometric info
    spall = fitsio.read(home+'Photo_dr16.fits.gz')
    ta = Table(spall)
    for gal in ['LOWZ']:#['LOWZ','CMASS']:
        print('matching the clustering-photometric data')
        for GC in ['North','South']:
            # clustering reading
            redrock = fitsio.read(home+'BOSS_data/galaxy_DR12v5_{}_{}.fits.gz'.format(gal,GC))
            tc = Table(redrock)
            # macthing photo and clustering
            tca = join(tc,ta, keys=['RUN', 'CAMCOL','ID','FIELD'],join_type='left')
            #import pdb;pdb.set_trace()
            #tca['flux_mag'] = tca['CMODELFLUX']*0
            #tca['gi'] = -2.5*(np.log10(tca['CMODELFLUX'][:,1]*tca['EXTINCTION_2'][:,3]/tca['CMODELFLUX'][:,3]/tca['EXTINCTION_2'][:,1]))
            tca['gi'] = tca['CMODELMAG'][:,1]-tca['CMODELMAG'][:,3]
            tca['gi_mod'] = tca['MODELMAG'][:,1]-tca['MODELMAG'][:,3]

            """
            for i in range(len(tca)):
                tca['flux055'][i] = kcorr(tca['Z'][i],tca['CMODELFLUX'][i],tca['CMODELFLUX_IVAR'][i])
                tca['gi'][i] = -2.5*np.log10(tca['flux055'][i][1]/tca['flux055'][i][3])
            """
            tca.write(filename.format(gal,GC), format='fits', overwrite=True)
else:
    for gal in ['LOWZ','CMASS']:
        print('colour study')
        # print the red/blue ratio
        if gal =='CMASS':
            zmins = [0.43,0.51,0.57,0.43]
            zmaxs = [0.51,0.57,0.7,0.7] 
        elif gal == 'LOWZ':
            zmins = [0.2, 0.33,0.2]
            zmaxs = [0.33,0.43,0.43]            
        hdu = fits.open(filename.format(gal,'North'))
        dataN = hdu[1].data;hdu.close()
        hdu = fits.open(filename.format(gal,'South')) 
        dataS = hdu[1].data;hdu.close()
        #weightN = dataN['WEIGHT_FKP']*dataN['WEIGHT_SYSTOT']*(dataN['WEIGHT_CP']+dataN['WEIGHT_NOZ']-1)
        #weightS = dataS['WEIGHT_FKP']*dataS['WEIGHT_SYSTOT']*(dataS['WEIGHT_CP']+dataS['WEIGHT_NOZ']-1
 
        # plot the color-redshift relations for LOWZ, CMASS and eBOSS
        ## clustering
        info = {'z_spec':[],'gi_spec':[],'w_spec':[],'z_rep':[],'gi_rep':[]}
        info['z_spec'] = np.append(dataN['Z'],dataS['Z'])
        #######################
        info['gi_spec'] = np.append(dataN['gi'],dataS['gi'])
        #######################
        #info['w_spec'] = np.append(weightN,weightS)
        #info['flux_spec'] = np.vstack((dataN['flux055'][selN],dataS['flux055'][selS]))
        info['cmodel_spec'] = np.vstack((dataN['CMODELMAG'],dataS['CMODELMAG']))
        info['model_spec'] = np.vstack((dataN['MODELMAG'],dataS['MODELMAG']))
        info['model_spec'] = np.vstack((dataN['MODELMAG'],dataS['MODELMAG']))
        info['psf_spec'] = np.vstack((dataN['PSFMAG'],dataS['PSFMAG']))
        info['fiber2_spec'] = np.vstack((dataN['FIBER2MAG'],dataS['FIBER2MAG']))
        dperp = info['model_spec'][:,2]-info['model_spec'][:,3]-(info['model_spec'][:,1]-info['model_spec'][:,2])/8
        cperp = info['model_spec'][:,2]-info['model_spec'][:,3]-(info['model_spec'][:,1]-info['model_spec'][:,2])/4-0.18
        cpara = 0.7*(info['model_spec'][:,1]-info['model_spec'][:,2])+1.2*(info['model_spec'][:,2]-info['model_spec'][:,3]-0.18)
        #import pdb;pdb.set_trace()
        sel = (info['z_spec']>zmins[-1])&(info['z_spec']<zmaxs[-1])
        simplecut = (info['gi_spec']<1e4)&sel

        if gal == 'CMASS': 
            sel &= (17.5<info['cmodel_spec'][:,3])&(info['cmodel_spec'][:,3]<19.9)
            sel &= dperp>0.55
            sel &= (info['model_spec'][:,2]-info['model_spec'][:,3])<2
            sel &= info['fiber2_spec'][:,3]<21.5
            sel &= (info['cmodel_spec'][:,3]<19.86+1.6*(dperp-0.8))
            sel &= (info['psf_spec'][:,3]-info['model_spec'][:,3])>0.2+0.2*(20-info['model_spec'][:,3])
            sel &= (info['psf_spec'][:,4]-info['model_spec'][:,4])>9.125-0.46*info['model_spec'][:,4]
        else:
            sel &= (16<info['cmodel_spec'][:,2])&(info['cmodel_spec'][:,2]<19.6)
            sel &= abs(cperp)<0.2
            sel &= (info['psf_spec'][:,2]-info['cmodel_spec'][:,2])>0.3
        title = 'selected with target selection criteria'
        
        # print the red/blue fraction
        for zmin,zmax in zip(zmins,zmaxs):
            for types in ['no selection','selected']:
                lentot = len(info['z_spec'][simplecut])
                if types =='selected':
                    lenblue = len(info['z_spec'][sel&colcut(info)])
                    lenred = len(info['z_spec'][sel&(~colcut(info))])
                else:
                    lenblue = len(info['z_spec'][colcut(info)])
                    lenred = len(info['z_spec'][~(colcut(info))])
                print('{} {} galaxies in {} at {}<z<{}, red = {} ({:.1f}%), blue = {}({:.1f}%)'.format(lentot,types,gal,zmin,zmax,lenred,lenred/lentot*100,lenblue,lenblue/lentot*100))
            """
                if types == 'selected':
                    selN = sel[:len(dataN)]&(zsel(dataN))
                    selS = sel[len(dataN):]&(dataS['Z']<zmax)&(dataS['Z']>=zmin) 
                else:
                    selN = (dataN['Z']<zmax)&(dataN['Z']>=zmin)
                    selS = (dataS['Z']<zmax)&(dataS['Z']>=zmin) 
                lentot = len(dataN[selN])+len(dataS[selS])
                lenred = len(dataN[selN&(dataN['gi']>=2.35)])+len(dataS[selS&(dataS['gi']>=2.35)])
                lenblue= len(dataN[selN&(dataN['gi']<2.35)])+len(dataS[selS&(dataS['gi']<2.35)])
                print('{} {} galaxies in {} at {}<z<{}, red = {} ({:.1f}%), blue = {}({:.1f}%)'.format(lentot,types,gal,zmin,zmax,lenred,lenred/lentot*100,lenblue,lenblue/lentot*100))
                
        
            lentotw = sum(weightN[selN])+sum(weightS[selS])
            lenredw = sum(weightN[selN&(dataN['gi']>=2.35)])+sum(weightS[selS&(dataS['gi']>=2.35)])
            lenbluew= sum(weightN[selN&(dataN['gi']<2.35)])+sum(weightS[selS&(dataS['gi']<2.35)])
            print('{:.1f} weighted galaxies in {} at {}<z<{}, red = {:.1f} ({:.1f}%), blue = {:.1f}({:.1f}%)'.format(lentotw,gal,zmin,zmax,lenredw,lenredw/lentotw*100,lenbluew,lenbluew/lentotw*100))
            """
        # the colour-redshift relation with/without cut
        fig, axs = plt.subplots(ncols=2,sharey=True,figsize=(12, 5))#
        #fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax = axs[0]        
        print(len(info['z_spec'][simplecut]),zmin,zmax)
        hb = ax.hexbin(info['z_spec'][simplecut],info['gi_spec'][simplecut],cmap='inferno',reduce_C_function=np.sum,gridsize=250,vmin=0)#,vmax=120)#info['w_spec'],
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')
        if gal == 'CMASS':
            ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.2,'w--',label='(g-i)=2.35')
        else:
            ax.plot(np.linspace(zmin,zmax,10),1.1+2.8*np.linspace(zmin,zmax,10),'w--',label='(g-i)=2.8z+1.1')
        #import pdb;pdb.set_trace()
        
        import matplotlib.tri as tri
        xi = np.linspace(zmin,zmax,51)
        yi = np.linspace(1.5,3.5,51)
        triang = tri.Triangulation(hb.get_offsets()[:,0],hb.get_offsets()[:,1])
        interpolator = tri.LinearTriInterpolator(triang, hb.get_array())
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)
        #ax.contourf(Xi,Yi,zi,colors='b',levels=2)
        ax.contour(Xi,Yi,zi,colors='b',levels=5)

        """
        import seaborn as sns
        sns.jointplot(info['z_spec'][simplecut],info['gi_spec'][simplecut], kind='kde', color="skyblue")
        """
        ax.set_title('no selection')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        ax.set_xlabel('z')
        ax.set_ylabel('(g-i)')

        ax = axs[1]     
        print(len(info['z_spec'][sel]))
        hb = ax.hexbin(info['z_spec'][sel],info['gi_spec'][sel],cmap='inferno',reduce_C_function=np.sum,gridsize=250,vmin=0)#,vmax=10)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')
        if gal == 'CMASS':
            ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.2,'w--',label='(g-i)=2.35')
        else:
            ax.plot(np.linspace(zmin,zmax,10),1.1+2.8*np.linspace(zmin,zmax,10),'w--',label='(g-i)=2.8z+1.1')

        triang = tri.Triangulation(hb.get_offsets()[:,0],hb.get_offsets()[:,1])
        interpolator = tri.LinearTriInterpolator(triang, hb.get_array())
        zi = interpolator(Xi, Yi)
        ax.contour(Xi,Yi,zi,colors='b',levels=5)

        ax.set_title(title)#('i-band selection')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        ax.set_xlabel('z')
        ax.set_ylabel('(g-i)')

        plt.legend(loc=0)
        plt.savefig('{}_colour_split-selection_mag.png'.format(gal))
        plt.close() 
"""
        # flux in cmodel ugriz   
        fig, axs = plt.subplots(ncols=5,nrows=2,figsize=(30, 10))
        for i,band in enumerate(['u','g','r','i','z']):
            hb = axs[0,i].hexbin(info['z_spec'],np.log10(info['cmodel_spec'])[:,i],cmap='YlGnBu',gridsize=200,reduce_C_function=np.sum,vmin=0,vmax=50)
            cb = fig.colorbar(hb, ax=axs[0,i])
            cb.set_label('counts')
            #plt.ylim(1.5,3.5)
            axs[0,i].set_xlim(zmin,zmax)
            axs[0,i].set_xlabel('redshift z')
            axs[0,i].set_ylabel('log10(flux)')
            axs[0,i].set_title('{} band cmodelflux'.format(band))
            
            hb = axs[1,i].hexbin(info['z_spec'][abn],np.log10(info['cmodel_spec'][abn])[:,i],cmap='YlGnBu',gridsize=200,reduce_C_function=np.sum,vmin=0,vmax=5)
            cb = fig.colorbar(hb, ax=axs[1,i])
            cb.set_label('counts')
            #plt.ylim(1.5,3.5)
            axs[1,i].set_xlim(zmin,zmax)
            axs[1,i].set_xlabel('redshift z')
            axs[1,i].set_ylabel('log10(flux)')
            axs[1,i].set_title('{} band cmodelflux for g-i=2.708'.format(band))
        plt.savefig('{}_cmodelflux_distr.png'.format(gal))
        plt.close()
"""              
"""
        ## repetitive observations
        datadir = '/global/homes/j/jiaxi/Vsmear-photo'
        filename = '{}/{}-{}_deltav_z{}z{}-{}.fits.gz'.format(datadir,'BOSS',gal,zmin,zmax,'NGC+SGC')
        hdu = fits.open(filename)
        data = hdu[1].data;hdu.close()        
        info['z_rep'] = data['z']*1
        info['gi_rep'] = data['gi']*1
        #import pdb;pdb.set_trace()
        ## hexbin for the colour-redshift relation
        fig, axs = plt.subplots(ncols=2,sharey=True,figsize=(12, 5))
        #fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax = axs[0]        
        hb = ax.hexbin(info['z_spec'],info['gi_spec'],cmap='YlGnBu',reduce_C_function=np.sum,gridsize=200,vmin=0,vmax=120)#info['w_spec'],
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('weighted counts')
        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w--',label='$^{{0.55}}$(g-i)=2.35')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        
        ax = axs[1]        
        hb = ax.hexbin(info['z_rep'],info['gi_rep'],cmap='YlGnBu',gridsize=200,reduce_C_function=np.sum,vmin=0,vmax=10)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')
        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w--',label='$^{{0.55}}$(g-i)=2.35')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        plt.legend(loc=0)
        plt.savefig('{}_colour_split.png'.format(gal))
        plt.close()
        
        # the colour-redshift relation with/without cut
        fig, axs = plt.subplots(ncols=2,sharey=True,figsize=(12, 5))
        #fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax = axs[0]        
        hb = ax.hexbin(info['z_spec'],info['gi_spec'],cmap='YlGnBu',reduce_C_function=np.sum,gridsize=200,vmin=0,vmax=120)#info['w_spec'],
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('weighted counts')
        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w--',label='$^{{0.55}}$(g-i)=2.35')
        ax.set_title('no selection')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        
        ax = axs[1]        
        print(len(info['z_spec']),len(info['z_spec'][sel]))
        hb = ax.hexbin(info['z_spec'][sel],info['gi_spec'][sel],cmap='YlGnBu',reduce_C_function=np.sum,gridsize=200,vmin=0,vmax=10)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')
        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w--',label='$^{{0.55}}$(g-i)=2.35')
        ax.set_title('i-band selection')
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        
        
        plt.legend(loc=0)
        plt.savefig('{}_colour_split-selection.png'.format(gal))
        plt.close()    
"""
