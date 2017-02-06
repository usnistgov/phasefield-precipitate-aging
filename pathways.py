
# coding: utf-8

# Overlay phase-field simulation compositions on ternary phase diagram

# Numerical libraries
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

# Runtime / parallel libraries
import glob, time, warnings
from itertools import chain
from multiprocessing import Pool


# Visualization libraries
import matplotlib.pylab as plt

from CALPHAD_energies import *


# # Generate a phase diagram
# Using scipy.spatial.ConvexHull, an interface to qhull. This method cannot provide phase fractions, chemical potentials, etc., but will quickly produce the correct diagram from the given Gibbs energies.

labels = [r'$\gamma$', r'$\delta$', r'$\mu$', 'LavesHT', 'LavesLT', 'BCC']
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

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
    a = n / density # index along x-axis
    b = n % density # index along y-axis

    xnb = epsilon + 1.0*a / (density-1)
    xcr = epsilon + 1.0*b / (density-1)
    xni = 1.0 - xcr - xnb

    result = [0]*7
    
    if xni>0:
        result[0] = xcr
        result[1] = xnb
        result[2] = xni
        result[3] = GG(xcr, xnb)
        result[4] = GD(xcr, xnb)
        result[5] = GU(xcr, xnb)
        result[6] = GL(xcr, xnb)
    
    return result


density = 101
allCr = []
allNb = []
allG = []
allID = []
points = []
phases = []

if __name__ == '__main__':
    pool = Pool(2)

    i = 0
    for result in pool.imap(computeKernelExclusive, range(density**2)):
        xcr, xnb, xni, fg, fd, fu, fh = result
        f = (fg, fd, fu, fh)

        # Accumulate (x, y, G) points for each phase
        if (fd**2 + fu**2 + fh**2) > epsilon:
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



for pair in ('collapse', 'gamma-delta', 'gamma-mu', 'gamma-laves', 'fourphase'):
    # Plot phase diagram
    plt.figure(figsize=(10, 7.5)) # inches
    plt.plot(XS, YS, '-k')
    plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
    plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
    plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
    n = 0
    for i in range(len(labels)):
        plt.scatter(X[i], Y[i], color=colors[i], s=2.5, label=labels[i])
    plt.xticks(np.linspace(0, 1, 21))
    plt.scatter(Xtick, Ytick, color='black', s=3)
    plt.legend(loc='best')
    
    # Add composition pathways
    fnames = glob.glob("data/alloy625/run1/{0}/*.xy".format(pair))
    n = len(fnames)
    for i in np.logspace(0, np.log10(n), 20, endpoint=False, dtype=int):
        xcr, xnb = np.loadtxt(fnames[i], delimiter=',', unpack=True)
        plt.plot(simX(xnb, xcr), simY(xcr), '.-', markersize=2, linewidth=1)
    
    plt.xlim([0, 0.6])
    plt.ylim([0, rt3by2*0.6])
    plt.savefig("diagrams/pathways_{0}.png".format(pair), dpi=600)
    plt.close()
