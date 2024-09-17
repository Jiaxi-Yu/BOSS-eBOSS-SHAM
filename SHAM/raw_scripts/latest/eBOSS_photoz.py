import numpy as np
from glob import glob
from astropy.io import fits
import fitsio
from astropy.table import Table, join
import numpy.ma as ma
from multiprocessing import Pool
import time
import pylab as plt
import sys
import re
import os

task = sys.argv[1]
home = '/global/cscratch1/sd/jiaxi/SHAM/catalog/'
print(task)
if task == 'plot':
    hdu = fitsio.read('eboss-decals_5e-4.fits.gz')
    data_woi = Table(hdu)
    unsel = np.isnan(data_woi['Z_PHOT_MEAN'])&(data_woi['Z_PHOT_MEAN']==-1000)&(data_woi['Z_PHOT_MEAN']==-99)
    countsc,bins = np.histogram(data_woi['Z_PHOT_MEAN'],bins = np.linspace(0.6,1.0,20))
    countsi,bins = np.histogram(data_woi['Z_PHOT_MEAN'][data_woi['i_FLAG']==0],bins = bins)
    # plot the completeness
    plt.rc('font', family='serif', size=15) 
    fig = plt.figure(figsize = (6,5),tight_layout=True)
    plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
    plt.plot((bins[1:]+bins[:-1])/2,countsi/countsc,'k',lw=2)
    plt.xlabel(r'$z_{{photo}}$')
    plt.ylabel('completeness')
    plt.ylim(0.82,1.02)
    plt.yticks([0.85,0.9,0.95,1.0])
    plt.xticks([0.6,0.7,0.8,0.9,1.0])
    #plt.savefig('completeness_eBOSS.pdf')
    plt.close()

    # plot the incompleteness to correspond to Vceil
    fig = plt.figure(figsize = (6,5),tight_layout=True)
    plt.subplots_adjust(left=0.1,right=0.9,top=0.9,bottom=0.1)
    plt.plot((bins[1:]+bins[:-1])/2,1-countsi/countsc,'k',lw=2)
    plt.xlabel(r'$z_{{photo}}$')
    plt.ylabel('incompleteness')
    plt.xlim(1.0,0.6)
    plt.savefig('incompleteness_eBOSS.png')
    plt.close()
elif task == 'mismatching':
    distance_thresh = 5e-4*2
    hdu = fitsio.read(home+'eboss-decals_5e-4.fits.gz')
    eboss = Table(hdu)  
    # set -1000 for long-distance matching; -99 from DeCaLS original data
    seldis = (eboss['Z_PHOT_MEAN']==-1000)
    selz   = (eboss['Z_PHOT_MEAN']==-99) 
    selobs = (np.isnan(eboss['Z_PHOT_MEAN']))
    print('{:.1f}% of eBOSS has no matching photo z'.format(100*len(eboss[seldis|selz|selobs])/len(eboss)))
    print('{:.1f}% of eBOSS has distance larger than {}'.format(100*len(eboss[seldis])/len(eboss),distance_thresh))
    print('{:.1f}% of eBOSS has no DECaLS photo z'.format(100*len(eboss[selz])/len(eboss)))
    print('{:.1f}% of eBOSS has no DECaLS bricks'.format(100*len(eboss[selobs])/len(eboss)))
    # filtering the mismatched one
    eboss = eboss[eboss['Z_PHOT_MEAN']==-1000]
    for i in range(3):
        np.savetxt('mismatching{}.csv'.format(i),np.array([eboss['RA'][20000*i:20000*(i+1)],eboss['DEC'][20000*i:20000*(i+1)]]).T,header='RA DEC')
elif task == 'brick-matching':
    distance_thresh = 5e-4*2
    # find the name of the sweep file
    def fname(ra, dec):
        ra1 = int((ra // 10) * 10)  
        dec1 = int((dec // 5) * 5) 
        sgn1 = int(dec >= 0) 
        ra2 = ra1 + 10 
        dec2 = dec1 + 5 
        sgn2 = int(dec +5 >= 0) 
        char = ['m','p']  
        return f'{ra1:03d}{char[sgn1]}{abs(dec1):03d}-{ra2:03d}{char[sgn2]}{abs(dec2):03d}'
    # read the eBOSS catalogue without i-band lower limit
    midver = home+'eBOSS_matching-progress/3.thresh_2e-5-to-5e-4/eboss-decals_5e-4.fits.gz'
    # 1.north-cover-south, 2.thresh_2e-5, 3.thresh_2e-5-to-5e-4, 
    finver = home+'eBOSS_matching-progress/final/eboss-decals_5e-4.fits.gz'
    ## if the intermidiate version doesn't exist, start from scratch
    if not os.path.exists(midver):
        print('no intermediate file')
        hdu = fitsio.read(home+'ebosstargets_woi.fits.gz',columns=['RA','DEC','i_FLAG'])
        eboss = Table(hdu)
        eboss['Z_PHOT_MEAN']= np.empty(len(eboss))
        eboss['RA_DECALS'], eboss['DEC_DECALS'] = np.empty(len(eboss)),np.empty(len(eboss))
    ## if the intermediate version exists, but not the final one
    ## checking the long-distance macthing from its output
    elif not os.path.exists(finver):
        print('no final file, start from ',midver)
        hdu = fitsio.read(midver)
        eboss = Table(hdu)
        eboss = eboss[eboss['Z_PHOT_MEAN']==-1000]
    ## if the intermediate the final version both exists
    else:
        print('matching has finished')
        exit()

    # find the corresponding decals tiles for eBOSS targets
    names = np.array([fname(eboss['RA'][i],eboss['DEC'][i]) for i in range(len(eboss))])
    names_uni,cnt = np.unique(names,return_counts=True)
    names_uni = names_uni[np.argsort(-cnt)]
    cnt = cnt[np.argsort(-cnt)]
    # matching eBOSS with DeCaLS
    for Fn,namep in enumerate(names_uni):
    #Fn=118;namep=names_uni[Fn]
    #if Fn>100:
        # the DeCaLS file in tile that contains several eBOSS target
        file_pzs = glob(home+'eBOSS_DeCaLS/*{}.fits.gz'.format(namep))
        # the eBOSS targets that is at a given tile
        sel = (names==namep)
        zmean = np.empty(len(eboss[sel]))
        rad,decd = np.empty(len(eboss[sel])),np.empty(len(eboss[sel]))

        if len(file_pzs)>0:
            # matching with each (RA,DEC) from eboss
            ## if eBOSS targets to match is small, use sequential
            if cnt[Fn]<63:
                # matching the eboss targets one by one sequentially
                for rd in range(len(eboss['RA'][sel])):
                    ra,dec = eboss['RA'][sel][rd],eboss['DEC'][sel][rd]
                    # read the selected decals files for matching
                    for pzi,file_pz in enumerate(file_pzs):
                        hdu = fitsio.read(file_pz)
                        decals = Table(hdu)
                        distance = np.sqrt((ra-decals['RA'])**2+(dec-decals['DEC'])**2)
                        # initialise the matching parameter
                        if pzi == 0:
                            matched = False
                        # if the min(eboss-decals distance)< distance threshold
                        # save the corresponding z_photo, RA, DEC
                        if distance[np.argmin(distance)]<distance_thresh:
                            zmean[rd] = decals['Z_PHOT_MEAN'][np.argmin(distance)]
                            rad[rd],decd[rd] = decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)]
                            matched = True
                            if len(file_pzs)==2:
                                print('matched,     z_phot={} in {} {}'.format(zmean[rd],rad[rd],decd[rd]))
                        # if the min(eboss-decals distance) > distance threshold
                        # save z_photo = -1000, and the closest, RA, DEC
                        else:
                            ## just to avoid the second file covers the first results 
                            if matched:
                                print('found again, z_phot={} in {} {}'.format(eboss[sel][rd]['Z_PHOT_MEAN'],eboss[sel][rd]['RA'],eboss[sel][rd]['DEC']))
                            else:
                                print('eBOSS {},{} for decals {},{} in {}, the smallest distance is larger than {}'.format(eboss['RA'][sel][rd],eboss['DEC'][sel][rd],decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)],namep,distance_thresh))                                
                                zmean[rd] = -1000
                                rad[rd],decd[rd] = decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)]
                # save the matched decals information
                eboss['Z_PHOT_MEAN'][sel]   = zmean.copy()
                eboss['RA_DECALS'][sel]     = rad.copy()
                eboss['DEC_DECALS'][sel]    = decd.copy()
            else:
            ## if there are many to match, parallelize
                def match(rd1):
                    ra,dec = eboss['RA'][sel][rd1],eboss['DEC'][sel][rd1]
                    for pzi,file_pz in enumerate(file_pzs):
                        hdu = fitsio.read(file_pz)
                        decals = Table(hdu)
                        distance = np.sqrt((ra-decals['RA'])**2+(dec-decals['DEC'])**2)
                        if pzi == 0:
                            matched = False
                        if distance[np.argmin(distance)]<distance_thresh:
                            zmean = decals['Z_PHOT_MEAN'][np.argmin(distance)]
                            rad,decd = decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)]
                            if matched:
                                print('2 matches in north and south')
                            matched = True
                            if len(file_pzs)==2:
                                print('matched,     z_phot={} in {} {}'.format(zmean,rad,decd))
                        else:
                            if matched:
                                print('found again, z_phot={} in {} {}'.format(zmean,rad,decd))
                                #.format(eboss[sel][rd1]['Z_PHOT_MEAN'],eboss[sel][rd1]['RA'],eboss[sel][rd1]['DEC']))
                                #zmean,rad,decd = eboss[sel][rd1]['Z_PHOT_MEAN'],eboss[sel][rd1]['RA'],eboss[sel][rd1]['DEC']
                            else:
                                print('eBOSS {},{} for decals {},{} in {}, the smallest distance is larger than {}'.format(ra,dec,decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)],namep,distance_thresh))                                
                                zmean = -1000
                                rad,decd = decals['RA'][np.argmin(distance)],decals['DEC'][np.argmin(distance)]                            
                    return [rd1,zmean,rad,decd]

                DATA = [0]*len(eboss[sel])
                # matching ra,dec in parallel for eboss targets in one sweep file
                pool = Pool()     
                for f, match_array in enumerate(pool.imap(match,np.arange(cnt[Fn]))):
                    DATA[f] = match_array
                pool.close() 
                pool.join()  
                ind = np.array([DATA[i][0] for i in range(cnt[Fn])])
                eboss['Z_PHOT_MEAN'][sel]   = np.array([DATA[i][1] for i in range(cnt[Fn])])[np.argsort(ind)]
                eboss['RA_DECALS'][sel]     = np.array([DATA[i][2] for i in range(cnt[Fn])])[np.argsort(ind)]
                eboss['DEC_DECALS'][sel]    = np.array([DATA[i][3] for i in range(cnt[Fn])])[np.argsort(ind)]
        else:
            # how many eboss samples has no decals measurements
            print('eboss',namep,'has no decals observation')
            eboss['Z_PHOT_MEAN'][sel] = np.nan
            eboss['RA_DECALS'][sel] = np.nan
            eboss['DEC_DECALS'][sel] = np.nan     
        #print(eboss[sel]['Z_PHOT_MEAN'])
    if not os.path.exists(midver):
        eboss.write(home+'eboss-decals_5e-4.fits.gz',format='fits', overwrite=True)
    elif not os.path.exists(finver):
        hdu = fitsio.read(midver)
        ebosstot = Table(hdu)
        ebosstot[ebosstot['Z_PHOT_MEAN']==-1000] = eboss
        ebosstot.write(home+'eboss-decals_5e-4.fits.gz',format='fits', overwrite=True)

        # set -1000 for long-distance matching; -99 from DeCaLS original data
        seldis = (ebosstot['Z_PHOT_MEAN']==-1000)
        selz   = (ebosstot['Z_PHOT_MEAN']==-99) 
        selobs = (np.isnan(ebosstot['Z_PHOT_MEAN']))
        print('{:.1f}% of eBOSS has no matching photo z'.format(100*len(ebosstot[seldis|selz|selobs])/len(ebosstot)))
        print('{:.1f}% of eBOSS has distance larger than {}'.format(100*len(ebosstot[seldis])/len(ebosstot),distance_thresh))
        print('{:.1f}% of eBOSS has no DECaLS photo z'.format(100*len(ebosstot[selz])/len(ebosstot)))
        print('{:.1f}% of eBOSS has no DECaLS bricks'.format(100*len(ebosstot[selobs])/len(ebosstot)))

elif task == 'eboss-matching':
    hdu = fitsio.read(home+'eBOSS_TS/output/ebosstarget/v0005/ebosstarget-v0005-lrg.fits',\
    columns=['RA','DEC'])
    data_wi = Table(hdu)
    hdu = fitsio.read(home+'eBOSS_TS_woi/output/ebosstarget/v0005/ebosstarget-v0005-lrg.fits')
    data_woi = Table(hdu)
    tac = join(data_wi,data_woi, keys=['RA'], join_type='right',table_names=['1','2'])
    test = ma.array(tac['DEC_1']) 
    tac['DEC_1'][np.where(test.mask==False)]=0 # appear in the final catalogue
    tac['DEC_1'][np.where(test.mask==True)]=1  # removed from the final
    tac['DEC_1'].name = 'i_FLAG'
    tac['DEC_2'].name = 'DEC'
    tac.write(home+'ebosstargets_woi.fits.gz',format='fits', overwrite=True)

elif task == 'decals-matching':
    # merge (RA, DEC) and photoz
    def merge(filename):    
        if not os.path.exists(home+'eBOSS_DeCaLS/{}_{}.gz'.format(filename[56:61],filename[-26:])):
            hdu = fitsio.read(filename,columns=['RA','DEC','OBJID'])
            datapos = Table(hdu)
            hdu = fitsio.read(filename[:-27]+'-photo-z'+filename[-27:-5]+'-pz.fits',columns=['OBJID','Z_PHOT_MEAN', 'Z_PHOT_MEDIAN'])
            datapz = Table(hdu)
            if np.all(datapz['OBJID']==datapos['OBJID'])==True:
                a = time.time()
                cols = [] 
                cols.append(fits.Column(name='RA',format='D',array=datapos['RA'])) 
                cols.append(fits.Column(name='DEC',format='D',array=datapos['DEC'])) 
                cols.append(fits.Column(name='Z_PHOT_MEAN',format='D',array=datapz['Z_PHOT_MEAN'])) 
                cols.append(fits.Column(name='Z_PHOT_MEDIAN',format='D',array=datapz['Z_PHOT_MEAN'])) 
                hdulist = fits.BinTableHDU.from_columns(cols) 
                hdulist.writeto(home+'eBOSS_DeCaLS/{}_{}.gz'.format(filename[56:61],filename[-26:]),overwrite=True)
                print('time cost: {:.1f}s'.format(time.time()-a))
            else:
                print('error:',filename)
        else:
            print(filename,' exists')
        # merger too slow
        #tac = join(datapos,datapz, keys=['OBJID'], join_type='left')
        #tac.write(home+'eBOSS_DeCaLS/{}.gz'.format(filename[-26:]),format='fits', overwrite=True)
    # file root
    filenames = glob('/global/project/projectdirs/cosmo/data/legacysurvey/dr9/*/sweep/9.0/*.fits')
    pool = Pool()     
    for f, merger_array in enumerate(pool.imap(merge,filenames)):
        merger_array
        if f==0:
            print('start merging')
        elif (f+1)%(np.ceil(len(filenames)/10))==0:
            print('{}\% merging finished'.format((f+1)//(len(filenames)/100)))
    pool.close() 
    pool.join()
elif task == 'colour-plot':
    # read the SDSS catalogue before target selection
    hdu = fits.open('./eBOSS_photoz/ebosstarget-v0005-lrg.fits')
    SDSS = hdu[1].data
    hdu.close()
    # sort the catalogue by RA in an ascending sequence
    data = SDSS[SDSS['RA'].argsort()]
    # flux to be used in the target selection
    r = np.array([data[i]['MODELMAG'][2] for i in range(len(data))])
    i = np.array([data[i]['MODELMAG'][3] for i in range(len(data))])
    W1 = data['W1_MAG']+2.699
    print('preliminary test: colour diagram')
    plt.plot(r-i,r-W1,'.',markersize=1)
    plt.ylim(-2,15);plt.ylabel('$r-W1$')
    plt.xlim(0,2.5);plt.xlabel('$r-i$')
    plt.axvline(0.98,color='k',label='$r-i=0.98$') 
    plt.plot(np.linspace(0,2.5,2),np.linspace(0,2.5,2)*2,'k',label='$r-W1 = 2(r-i)$')
    plt.title('ebosstarget-v0005-lrg.fits')
    plt.legend(loc=0)
    plt.savefig('fig1_before-target-selection.png')
    plt.close() 

    hdu = fits.open('./eBOSS_TS/eBOSS_LRG_full_ALLdata-vDR16.fits')
    data = hdu[1].data
    hdu.close()
    r = np.array([data[i]['MODELMAG'][2] for i in range(len(data))])
    i = np.array([data[i]['MODELMAG'][3] for i in range(len(data))])
    W1 = data['W1_MAG']
    plt.plot(r-i,r-W1,'.',markersize=1)
    plt.ylim(-2,15);plt.ylabel('$r-W1$')
    plt.xlim(0,2.5);plt.xlabel('$r-i$')
    plt.axvline(0.98,color='k',label='$r-i=0.98$') 
    plt.plot(np.linspace(0,2.5,2),np.linspace(0,2.5,2)*2,'k',label='$r-W1 = 2(r-i)$')
    plt.legend(loc=0)
    plt.title('eBOSS_LRG_full_ALLdata-vDR16.fits')
    plt.savefig('fig1_final.png')
    plt.close() 

elif task == 'TS':
    files = glob('/global/cscratch1/sd/zhaoc/tmp/ebosstarget/output/ebosstarget/v0005/tmp/ebosstarget-lrg-*.fits')
    hdu = fits.open(files[0])
    data = hdu[1].data
    #SDSS = hdu[1].data
    hdu.close()
    # sort the catalogue by RA in an ascending sequence
    #data = SDSS[SDSS['RA'].argsort()]
    # flux to be used in the target selection
    r = np.array([data[i]['MODELMAG'][2] for i in range(len(data))])
    i = np.array([data[i]['MODELMAG'][3] for i in range(len(data))])
    W1 = data['W1_MAG']+2.699

    for j in range(1,len(files)):
        hdu = fits.open(files[j])
        data = hdu[1].data
        #SDSS = hdu[1].data
        hdu.close()
        # sort the catalogue by RA in an ascending sequence
        #data = SDSS[SDSS['RA'].argsort()]
        # flux to be used in the target selection
        r = np.append(r,np.array([data[i]['MODELMAG'][2] for i in range(len(data))]))
        i = np.append(i , np.array([data[i]['MODELMAG'][3] for i in range(len(data))]))
        W1= np.append(W1,data['W1_MAG']+2.699 )

    plt.plot(r-i,r-W1,'.',markersize=1)
    plt.ylim(-2,15);plt.ylabel('$r-W1$')
    plt.xlim(0,2.5);plt.xlabel('$r-i$')
    plt.axvline(0.98,color='k',label='$r-i=0.98$') 
    plt.plot(np.linspace(0,2.5,2),np.linspace(0,2.5,2)*2,'k',label='$r-W1 = 2(r-i)$')
    plt.legend(loc=0)
    plt.title('appended catalogues after target selection')
    plt.savefig('fig1_new.png')
    plt.close()    


"""
# test: determine the bricknames of each galaxy
from desiutil.brick import brickname
hdu = fits.open(home+'eBOSS_clustering_fits/eBOSS_LRG_clustering_NGC_v7_2.dat.fits')
dataN = hdu[1].data
hdu.close()
hdu = fits.open(home+'eBOSS_clustering_fits/eBOSS_LRG_clustering_SGC_v7_2.dat.fits')
dataS = hdu[1].data
hdu.close()
data = np.append(dataN,dataS)
# identify all the bricks
brickID = brickname(data['RA'],data['DEC'])
np.savetxt('bricknames.txt',brickID, fmt='%s')
# identify the initials of the bricks
brickini = [int(i[:3]) for i in brickID] 
np.unique(brickini)

with open('bricknames.txt', 'r') as td:
    for line in td:
        info = re.split(' +', line)
        columns.append(info[0][:-1])
"""