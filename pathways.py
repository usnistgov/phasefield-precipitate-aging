# coding: utf-8

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.

# Usage: python pathways.py data/alloy625/run2/TKR4p119/run2*

# Numerical libraries
import re
import numpy as np
from math import floor, sqrt
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

# Runtime / parallel libraries
from os import path, stat
from sys import argv

import glob, time
from itertools import chain
from multiprocessing import Pool

# Visualization libraries
import matplotlib.pylab as plt


density = 500
skipsz = 9

# # Generate a phase diagram
# Using scipy.spatial.ConvexHull, an interface to qhull. This method cannot provide phase fractions, chemical potentials, etc., but will quickly produce the correct diagram from the given Gibbs energies.

from CALPHAD_energies import *

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

# Check data directory

for j in range(1, len(argv)):
    datdir = argv[j]
    if path.isdir(datdir) and len(glob.glob("{0}/*.xy".format(datdir))) > 0:
        base = path.basename(datdir)

        # Plot phase diagram
        plt.figure(0, figsize=(10, 7.5)) # inches
        plt.plot(XS, YS, '-k')
        plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
        plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=18)
        plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=18)
        for i in range(len(labels)):
            plt.scatter(X[i], Y[i], color=colors[i], s=2, label=labels[i])
            plt.scatter(X0[i], Y0[i], color='black', s=6, zorder=10)
        plt.xticks(np.linspace(0, 1, 21))
        plt.scatter(Xtick, Ytick, color='black', s=3)

        plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
        plt.text(simX(0.230, 0.010), simY(0.010), r'$\delta$', fontsize=14)
        plt.text(simX(0.310, 0.320), simY(0.320), r'L',        fontsize=14)

        plt.figure(1, figsize=(10, 7.5)) # inches
        plt.title("Driving Force for Transformation", fontsize=18)
        plt.xlabel(r'$x$', fontsize=18)
        plt.ylabel(r'$P$', fontsize=18)

        # Add composition pathways
        fnames = sorted(glob.glob("{0}/*.xy".format(datdir)))
        n = len(fnames)
        for i in np.arange(0, n, min(n,skipsz), dtype=int):
            x, xcr, xnb, P = np.loadtxt(fnames[i], delimiter=',', unpack=True)
            num = int(re.search('[0-9]{5,16}', fnames[i]).group(0)) / 1000000
            plt.figure(0)
            plt.plot(simX(xnb, xcr), simY(xcr), '.-', markersize=2, linewidth=1, zorder=1, label=num)
            plt.figure(1)
            plt.semilogy(x, P, label=num)

        plt.figure(0)
        plt.xlim([0, 0.6])
        plt.ylim([0, rt3by2*0.6])
        plt.legend(loc='best')
        plt.savefig("diagrams/pathways_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()

        plt.figure(1)
        plt.legend(loc='best', fontsize=8)
        plt.savefig("diagrams/pressures_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()

        # Plot phase evolution trajectories
        t, fd, fl, fg = np.loadtxt("{0}/phasefrac.csv".format(datdir), delimiter=',', skiprows=1, unpack=True)
        plt.figure(2, figsize=(10, 7.5)) # inches
        plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'Phase fraction $\phi$', fontsize=18)
        #plt.scatter(t, fg, c=colors[0], label="$\gamma$")
        plt.scatter(t, fd, c=colors[1], label="$\delta$")
        plt.scatter(t, fl, c=colors[2], label="Laves")
        plt.xlim([0, 100e6])
        plt.ylim([0, 3e-13])
        plt.legend(loc='best')
        plt.savefig("diagrams/phasefrac_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()
    else:
        print("Invalid argument: {0} is not a directory, or contains no usable data.".format(datdir))
        print("Usage: {0} path/to/data".format(argv[0]))
