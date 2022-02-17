#!/usr/bin/env python3
from astropy.io import fits
from astropy.table import Table, join
import fitsio
import numpy as np
import pylab as plt
import os

home = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
datadir = '/global/homes/j/jiaxi/SDSS_redshift_uncertainty/Vsmear-photo/'

def LOWZcut(x):
    return 2.8*x+1.2

def colcut(DATA):
    if gal == 'CMASS':
        colsel = DATA['gi_spec']<2.35
    elif gal == 'LOWZ':
        colsel = DATA['gi_spec']<LOWZcut(DATA['z_spec'])
    return colsel

if not os.path.exists(datadir+'LOWZ_South_mag.fits.gz'):
    # photometric info
    spall = fitsio.read(home+'Photo_dr16.fits.gz')
    ta = Table(spall)
    for gal in ['LOWZ']:#['LOWZ','CMASS']:
        print('matching the clustering-photometric data')
        for GC in ['North','South']:
            # clustering reading
            filename = datadir+'{}_{}_mag.fits.gz'.format(gal,GC)
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
    GC = 'NGC+SGC'
    for gal in ['CMASS','LOWZ']:
        filename = datadir+'{}_{}_mag.fits.gz'
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
            # the repeat data
            filename = '{}/{}-{}_deltav_z{}z{}-{}.fits.gz'.format(datadir,'BOSS',gal,zmin,zmax,GC)
            hdu = fits.open(filename)
            repeat = hdu[1].data
            hdu.close()
            w = (repeat['delta_chi2'] > 9)&(abs(repeat['delta_v'])<1000)
            repeat = repeat[w]

            # clustering plot
            zsel = (info['z_spec']>zmin)&(info['z_spec']<zmax)
            TYPES = ['clustering','repeat']#['no selection','selected']:
            for types in TYPES:
                if types =='selected':
                    lentot = len(info['z_spec'][sel&zsel])
                    lenblue = len(info['z_spec'][sel&zsel&colcut(info)])
                    lenred = len(info['z_spec'][sel&zsel&(~colcut(info))])
                else:
                    lentot = len(info['z_spec'][simplecut&zsel])
                    lenblue = len(info['z_spec'][colcut(info)&zsel])
                    lenred = len(info['z_spec'][(~colcut(info))&zsel])
                print('{} {} galaxies in {} at {}<z<{}, red = {} ({:.1f}%), blue = {}({:.1f}%)'.format(lentot,types,gal,zmin,zmax,lenred,lenred/lentot*100,lenblue,lenblue/lentot*100))

        # the colour-redshift relation with/without cut
        fontsize =15
        plt.rc('text', usetex=False)
        plt.rc('font', family='serif', size=fontsize)
        fig, axs = plt.subplots(ncols=2,figsize=(10, 5),sharey=True)#
        fig.subplots_adjust(left=0.1, right=0.97)

        ax = axs[0]        
        hb = ax.hexbin(info['z_spec'][simplecut],info['gi_spec'][simplecut],cmap='inferno',reduce_C_function=np.sum,gridsize=(50,160),vmin=0,rasterized=True,linewidths=0.2)#,vmax=120)#info['w_spec'],
        cb = fig.colorbar(hb, ax=ax)
        #cb.set_label('counts')

        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w',label='$(g-i)=2.35$')
        if gal == 'CMASS':
            vmax=35
        else:
            ax.plot(np.linspace(zmin,zmax,10),LOWZcut(np.linspace(zmin,zmax,10)),'w--',label='$(g-i)=2.8z+1.2$')
            vmax=15
        
        # plot contours
        if types.find('select') !=-1:
            import matplotlib.tri as tri
            xi = np.linspace(zmin,zmax,51)
            yi = np.linspace(1.5,3.5,51)
            triang = tri.Triangulation(hb.get_offsets()[:,0],hb.get_offsets()[:,1])
            interpolator = tri.LinearTriInterpolator(triang, hb.get_array())
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            #ax.contourf(Xi,Yi,zi,colors='b',levels=2)
            ax.contour(Xi,Yi,zi,colors='b',levels=5)
        
        ax.set_title('{} catalogue'.format(TYPES[0]))
        ax.set_ylim(1.5,3.5)
        ax.set_xlim(zmin,zmax)
        ax.set_xlabel('redshift',fontsize=fontsize)
        ax.set_ylabel('$(g-i)$',fontsize=fontsize)

        ax = axs[1]     
        if types.find('select') !=-1:
            print(len(info['z_spec'][sel]))
            hb = ax.hexbin(info['z_spec'][sel],info['gi_spec'][sel],cmap='inferno',reduce_C_function=np.sum,gridsize=(60,140),vmin=0,rasterized=True,linewidths=0.2)#,vmax=10)
        else:
            print(len(repeat['z']))
            hb = ax.hexbin(repeat['z'],repeat['gi'],cmap='inferno',reduce_C_function=np.sum,gridsize=(60,140),vmin=0,vmax=vmax,rasterized=True,linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('counts')
        ax.plot(np.linspace(zmin,zmax,10),np.ones(10)*2.35,'w',label='$(g-i)=2.35$')
        if gal == 'LOWZ':
            ax.plot(np.linspace(zmin,zmax,10),LOWZcut(np.linspace(zmin,zmax,10)),'w--',label='$(g-i)=2.8z+1.2$')
        if types.find('select') !=-1:
            triang = tri.Triangulation(hb.get_offsets()[:,0],hb.get_offsets()[:,1])
            interpolator = tri.LinearTriInterpolator(triang, hb.get_array())
            zi = interpolator(Xi, Yi)
            ax.contour(Xi,Yi,zi,colors='b',levels=5)
        ax.set_title('{} catalogue'.format(TYPES[1]))
        plt.ylim(1.5,3.5)
        plt.xlim(zmin,zmax)
        ax.set_xlabel('z',fontsize=fontsize)
        #ax.set_ylabel('(g-i)',fontsize=fontsize-3)

        plt.legend(loc=0)
        plt.savefig('{}_colour_split-selection_mag-repeat.pdf'.format(gal),dpi=150)
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
