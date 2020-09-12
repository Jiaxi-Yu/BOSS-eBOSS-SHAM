import numpy as np
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
        mono[~mask]=(dd[i][~mask]-2*dr[i][~mask]+rr[i][~mask])/rr[i][~mask])
        quad = mono * 2.5 * (3 * mu**2 - 1)
        hexa = mono * 1.125 * (35 * mu**4 - 30 * mu**2 + 3)

        xi0[i] = np.sum(mono,axis=1)/nmu
        xi2[i] = np.sum(quad,axis=1)/nmu
        xi4[i] = np.sum(hexa,axis=1)/nmu

    return s, xi0, xi2, xi4
 