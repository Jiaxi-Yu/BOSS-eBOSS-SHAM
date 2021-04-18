import numpy as np
import re
import os
# for EZmocks
def read_cnt(ifile, ds=1, isdd=False, ns=200, nmu=120):
    d = np.loadtxt(ifile, unpack=True)
    if len(d[0]) != ns * nmu:
        raise ValueError('wrong size of pair count file')

    if ns % ds != 0:
        raise ValueError('wrong bin size of separation')
    sbin = int(ns / ds)
    cnt = np.sum(d[5].reshape([sbin, ds, nmu]), axis=1)

    if isdd:
        mu = (d[0] + d[1]) * 0.5
        mu = np.median(mu.reshape([sbin, ds, nmu]), axis=1)
        smin = d[2][0]
        smax = d[3][-1]
        se = np.linspace(smin, smax, sbin+1)
        s = (se[1:] + se[:-1]) * 0.5

        idx = d[5] != 0
        ndata = np.nanmean(np.sqrt(d[4][idx]/d[5][idx]))
        return s, mu, cnt, ndata

    return None, None, cnt, None
def read_xi(ifmt, rfmt=None, ds=1, ns=200, nmu=120):
    caps = ['NGC', 'SGC']
    dd = [None] * 3
    dr = [None] * 3
    rr = [None] * 3
    xi0 = [None] * 3
    xi2 = [None] * 3
    xi4 = [None] * 3
    num = [None] * 2

    for i,cap in enumerate(caps):
        ifile = ifmt.format(cap,'dd')
        s, mu, dd[i], num[i] = read_cnt(ifile, ds=ds, isdd=True, ns=ns, nmu=nmu)
        ifile = ifmt.format(cap,'dr')
        _, _, dr[i], _ = read_cnt(ifile, ds=ds, isdd=False, ns=ns, nmu=nmu)

        if rfmt == None: ifile = ifmt.format(cap,'rr')
        else: ifile = rfmt.format(cap)
        _, _, rr[i], _ = read_cnt(ifile, ds=ds, isdd=False, ns=ns, nmu=nmu)

    nfac = num[1] / num[0]
    dd[2] = (dd[0] + dd[1] * nfac**2) / (1 + nfac)**2
    dr[2] = (dr[0] + dr[1] * nfac**2) / (1 + nfac)**2
    rr[2] = (rr[0] + rr[1] * nfac**2) / (1 + nfac)**2

    for i in range(3):
        mask = (rr[i]==0)
        mono = np.zeros_like(rr[i])
        mono[~mask]=(dd[i][~mask]-2*dr[i][~mask]+rr[i][~mask])/rr[i][~mask]
        quad = mono * 2.5 * (3 * mu**2 - 1)
        hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)

        xi0[i] = np.sum(mono,axis=1)/nmu
        xi2[i] = np.sum(quad,axis=1)/nmu
        xi4[i] = np.sum(hexa,axis=1)/nmu

    return s, xi0, xi2, xi4
 
# FCFC paircounts
def FCFCcomb(ifmt,rfmt,ds=1, ns=100, nmu=120):
    if os.path.exists(ifmt.format('NGC+SGC','mps')):
        xi0 = [None] * 3
        xi2 = [None] * 3
        xi4 = [None] * 3
        caps = ['North','South','NGC+SGC']     
        for j,cap in enumerate(caps):
            smid,smin,smax,xi0[j],xi2[j],xi4[j] = np.loadtxt(ifmt.format(cap,'mps'),unpack = True)
    else:
        caps = ['NGC', 'SGC']
        dd = [None] * 3
        dr = [None] * 3
        rr = [None] * 3
        xi0 = [None] * 3
        xi2 = [None] * 3
        xi4 = [None] * 3
        num = [None] * 2
        norm = [None] * 2

        for i,cap in enumerate(caps):
            # dd
            ifile = ifmt.format(cap,'dd')
            Smin,Smax,MUmin,MUmax,dd[i] = np.loadtxt(ifile,unpack=True)
            # read the normalisation 
            COMMENT_CHAR = '#'
            columns = []
            with open(ifile, 'r') as td:
                for line in td:
                    if line[0] == COMMENT_CHAR:
                        info = re.split(' +', line)
                        columns.append(info)
            num[i]  = np.float64(columns[1][-2])
            norm[i] = np.float64(columns[2][-1]) 
            # dr,rr
            ifile = ifmt.format(cap,'dr')
            Smin,Smax,MUmin,MUmax,dr[i] = np.loadtxt(ifile,unpack=True)        
            ifile = rfmt.format(cap)
            Smin,Smax,MUmin,MUmax,rr[i] = np.loadtxt(ifile,unpack=True)
        mu = (MUmin+MUmax)/2
        # NGC+SGC
        dd[2] = dd[0] * norm[0] + dd[1] * norm[1]
        dr[2] = dr[0] * norm[0] + dr[1] * norm[1]
        rr[2] = rr[0] * norm[0] + rr[1] * norm[1]

        for i in range(3):
            mask = (rr[i]==0)
            mono = np.zeros_like(rr[i])
            mono[~mask]=(dd[i][~mask]-2*dr[i][~mask]+rr[i][~mask])/rr[i][~mask]
            quad = mono * 2.5 * (3 * mu**2 - 1)
            hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)

            xi0[i] = np.sum(mono.reshape(nmu,ns),axis=0)/nmu
            xi2[i] = np.sum(quad.reshape(nmu,ns),axis=0)/nmu
            xi4[i] = np.sum(hexa.reshape(nmu,ns),axis=0)/nmu
        smin = np.unique(Smin)
        smax = np.unique(Smax)
        smid = (smin+smax)/2
        np.savetxt(ifmt.format('NGC+SGC','xi'),np.array([Smin,Smax,MUmin,MUmax,mono]).T,header='weighted galaxy: NGC {}, SGC {} \n normalisation: NGC {}, SGC {}'.format(num[0],num[1],norm[0],norm[1]))
        np.savetxt(ifmt.format('NGC+SGC','mps'),np.array([smid,smin,smax,xi0[2],xi2[2],xi4[2]]).T,header='weighted galaxy: NGC {}, SGC {} \n normalisation: NGC {}, SGC {}'.format(num[0],num[1],norm[0],norm[1]))

    return xi0,xi2,xi4