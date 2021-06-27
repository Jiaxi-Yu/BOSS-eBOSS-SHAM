#!/usr/bin/env python

import numpy as np
from scipy.special import hyp2f1
from scipy.misc import derivative
from scipy.integrate import quad

c = 299792.458
h = 0.6777
Om = 0.307115
sigma8 = 0.8225

def D(a):
    return a * hyp2f1(1./3.,1,11./6.,a**3*(1-1./Om))

def f(a):
    derv = derivative(D, a, dx=1e-3)
    return a * derv / D(a)

def fs8(a):
    derv = derivative(D, a, dx=1e-3)
    return sigma8 / D(1) * a * derv

def G(a):
    return D(a) / D(1)

def s8(a):
    return G(a) * sigma8

def efunc(a):
    return np.sqrt(1 - Om + Om / a**3)

def hubble(a):
    return 100 * h * np.sqrt(1 - Om + Om / a**3)

int4dist = lambda x : 1 / (x**2 * efunc(x))
def com_dist(a):
    return quad(int4dist, a, 1)[0] * c / (100 * h)

def ang_dia_dist(a):
    return a * com_dist(a)


z = 0.86
a = 1/(1+z)
#print('f:',f(a))
#print('fs8:',fs8(a))
#print('sigma8:',s8(a))
#print('G:',G(a))
#print('hubble:',hubble(a))
#print('comoving distance:',com_dist(a))
#print('angular diameter distance:',ang_dia_dist(a))

'''
you may use this formula to check the effective redshift of the pk:
P(k,z) = Plin(k) * D1(z)^2 * b^2
where D1 is the growth factor.
'''
