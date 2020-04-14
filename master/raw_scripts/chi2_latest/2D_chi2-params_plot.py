
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

# double
file = glob.glob('E:/Master/OneDrive/master_thesis/master/chi2/ELG/chi2-double*.txt')
k,j=1,0
name = ['sigma_high','sigma_low','v_max']
a=np.loadtxt(file[k],unpack=True)
# multiple parameters
fig,ax = plt.subplots()
ax = plt.subplot(projection='3d')
ax.set_xlabel(name[j])
ax.set_ylabel(name[k])
#ax.set_zlim(63,67)
ax.scatter(a[j],a[k],a[-1],cmap='Reds')
plt.show()
#plt.savefig(file[:-4]+'.png',bbox_tight=True)

#triple
file = 'E:/Master/OneDrive/master_thesis/master/chi2/ELG/chi2-triple.txt'
name = ['sigma_high','sigma_low','v_max']
a=np.loadtxt(file,unpack=True)
# multiple parameters
for k in range(3):
    fig,ax = plt.subplots()
    ax = plt.subplot(projection='3d')
    ax.set_xlabel(name[k])
    ax.set_ylabel(name[(k+1)%3])
    #ax.set_zlim(63,67)
    ax.scatter(a[k],a[(k+1)%3],a[-1],cmap='Reds')
    plt.show()
#plt.savefig(file[:-4]+'.png',bbox_tight=True)