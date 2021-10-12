#!/usr/bin/env python3
import kcorrect

def kcorr(z,flux,fluxivar,zrecon=0.55):
    # the redshift of target; flux and fluxivar in nanomaggie; the reconstructed redhshift
    coeffs = kcorrect.fit_nonneg(z,flux*1e-9,fluxivar*1e18)
    #return in nanomaggie
    return kcorrect.reconstruct_maggies(coeffs,redshift=zrecon)[1:]/1e-9
