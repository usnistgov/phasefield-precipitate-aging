
# coding: utf-8

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.

# Numerical libraries
import numpy as np
from gc import collect
from math import floor, sqrt
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

# Runtime / parallel libraries
import glob, time
from itertools import chain
from multiprocessing import Pool


# Visualization libraries
import matplotlib.pylab as plt

from CALPHAD_energies import *


# # Generate a phase diagram
# Using scipy.spatial.ConvexHull, an interface to qhull. This method cannot provide phase fractions, chemical potentials, etc., but will quickly produce the correct diagram from the given Gibbs energies.

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    xni = 1.0 - xcr
    Xtick.append(xcr/2 - 0.002)
    Ytick.append(rt3by2*xcr)
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.0 - xcr
    Xtick.append(xnb + xcr/2 + 0.002)
    Ytick.append(rt3by2*xcr)


def computeKernelExclusive(n):
    #a = n / density # index along x-axis
    #b = n % density # index along y-axis

    xnb = max(epsilon, 1.0 * (n / density) / density)
    xcr = max(epsilon, 1.0 * (n % density) / density)

    xni = 1.0 - xcr - xnb

    result = [0] * 5
    
    if xni>0:
        result[0] = xcr
        result[1] = xnb
        result[2] = PG(xcr, xnb)
        result[3] = PD(xcr, xnb)
        result[4] = PL(xcr, xnb)
    
    return result


density = 2000
allCr = []
allNb = []
allG = []
allID = []
points = []
phases = []

if __name__ == '__main__':
    pool = Pool(6)

    i = 0
    for result in pool.imap(computeKernelExclusive, range(density*(density+1))):
        xcr, xnb, fg, fd, fl = result
        f = (fg, fd, fl)

        # Accumulate (x, y, G) points for each phase
        for n in range(len(f)):
            allNb.append(simX(xnb, xcr))
            allCr.append(simY(xcr))
            allG.append(f[n])
            allID.append(n)
        i += 1

    pool.close()
    pool.join()
    
    points = np.array([allNb, allCr, allG]).T
    
    hull = ConvexHull(points)


# Prepare arrays for plotting
X = [[],[],[],[], [], []]
Y = [[],[],[],[], [], []]

for simplex in hull.simplices:
    for i in simplex:
        X[allID[i]].append(allNb[i])
        Y[allID[i]].append(allCr[i])



pair = 'threephase'
# Plot phase diagram
plt.figure(figsize=(10, 7.5)) # inches
plt.plot(XS, YS, '-k')
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=18)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=18)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=2, label=labels[i])
    plt.scatter(X0[i], Y0[i], color='black', s=6, zorder=10)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)

plt.legend(loc='best')
plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
plt.text(simX(0.310, 0.320), simY(0.320), r'L',        fontsize=14)

# Add composition pathways
fnames = sorted(glob.glob("data/alloy625/run1/{0}/*.xy".format(pair)))
n = len(fnames)
# for i in np.logspace(0, np.log10(n), 3, endpoint=False, dtype=int):
for i in range(n):
    xcr, xnb = np.loadtxt(fnames[i], delimiter=',', unpack=True)
    xcr0 = np.mean(xcr)
    xnb0 = np.mean(xnb)
    plt.plot(simX(xnb, xcr), simY(xcr), '.-', markersize=2, linewidth=1, zorder=1)
    plt.scatter(simX(xnb0, xcr0), simY(xcr0), color='black', s=6, zorder=10)

plt.xlim([0, 0.6])
plt.ylim([0, rt3by2*0.6])
plt.savefig("diagrams/pathways_{0}.png".format(pair), dpi=400, bbox_inches='tight')
plt.close()
collect()
