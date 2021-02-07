import matplotlib 
matplotlib.use('agg')
import numpy as np
from numpy import log, pi,sqrt,cos,sin,argpartition,copy,trapz,float32,int32,append,mean,cov,vstack,hstack
from astropy.table import Table
import matplotlib.pyplot as plt

a= np.loadtxt('mps_linear_LRG_SGC_z0.6z1.0/multinest_.txt')          

sel = a[:,1]<50
mini = a[:,1]==min(a[:,1])
plt.scatter(a[:,2][sel],a[:,1][sel],s=2)
plt.scatter(a[:,2][mini],a[:,1][mini],marker='o',label='min $\chi^2$')
plt.xlabel('$\sigma$')
plt.ylabel('$\chi^2$')
plt.legend(loc=0)
plt.savefig('chi2-sigma.png')
plt.close()