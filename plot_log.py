#####################################################################################
# This software was developed at the National Institute of Standards and Technology #
# by employees of the Federal Government in the course of their official duties.    #
# Pursuant to title 17 section 105 of the United States Code this software is not   #
# subject to copyright protection and is in the public domain. NIST assumes no      #
# responsibility whatsoever for the use of this code by other parties, and makes no #
# guarantees, expressed or implied, about its quality, reliability, or any other    #
# characteristic. We would appreciate acknowledgement if the software is used.      #
#                                                                                   #
# This software can be redistributed and/or modified freely provided that any       #
# derivative works bear some notice that they are derived from it, and any modified #
# versions bear some notice that they have been modified.                           #
#####################################################################################

import matplotlib.pylab as plt
import numpy as np

def moving_average(x, n, type='simple'):
    """
    compute an n period moving average, from http://matplotlib.org/examples/pylab_examples/finance_work2.html

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


#xCr, xNb, pG, pD, pU, pL, f, fails = np.loadtxt('c.log',delimiter='\t',unpack=True)
#xCr, xNb, pG, pD, pU, pL, f, fails = np.genfromtxt('c.log', delimiter='\t', unpack=True, skip_footer=1)
xCr, xNb, pG, pD, pU, pL, f, fails = np.genfromtxt('/data/tnk10/phase-field/alloy625/run1/c.log', delimiter='\t', unpack=True, skip_footer=1)
n = len(f)
t = 100000 * np.linspace(0,n-1,n)

plt.plot(t, xCr, linewidth=5,color='blue', label='Cr')
plt.plot(t, xNb, linewidth=5,color='red', label='Nb')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$\Sigma x$',fontsize=28)
plt.legend(loc='best')
plt.savefig("mass.png", dpi=600, bbox_inches='tight')
plt.close()

#plt.plot(t, pG, linewidth=5,color='black', label='$\gamma$')
plt.plot(t, pD, linewidth=5,color='green', label='$\delta$')
plt.plot(t, pU, linewidth=5,color='blue', label='$\mu$')
plt.plot(t, pL, linewidth=5,color='red', label='Laves')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$\Sigma\phi$',fontsize=28)
plt.legend(loc='best')
plt.savefig("phase.png", dpi=600, bbox_inches='tight')
plt.close()

# Smooth data to avoid errors due to excessively spiky line
N = np.maximum(1, n/1000000)

f0 = 0.99 * np.min(f)
if f0 < 0.0:
    f0 *= 1.01 / 0.99

#y = moving_average(f-f0, N, 'simple')
plt.semilogy(t, f-f0, linewidth=5,color='k')
plt.xlabel('$t$',fontsize=28)
plt.ylabel('$\Delta\mathcal{F}$',fontsize=28)
plt.savefig("energy.png", dpi=600, bbox_inches='tight')
plt.close()

#y = moving_average(100.0*fails, N, 'simple')
plt.plot(t, 100.0*fails, linewidth=5)
plt.xlabel('$t$',fontsize=28)
plt.ylabel('Failures (%)',fontsize=28)
plt.savefig("fail.png", dpi=600, bbox_inches='tight')
plt.close()

