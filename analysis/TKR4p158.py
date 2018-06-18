#!/usr/bin/python
# -*- coding: utf-8 -*-

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

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.
# Usage: python analysis/TKR4p158.py

import glob
from os import path
import numpy as np
import matplotlib.pylab as plt
from sympy import Matrix, solve_linear_system, symbols

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

dt = 7.5e-5 * 1000
tmax = 5625 # 3750
pmax = 0.35

# Plot difference in size between delta and Laves precipitates
plt.figure(1, figsize=(10, 7.5)) # inches
plt.title("Cr-Nb-Ni at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$t$ / s', fontsize=18)
plt.ylabel(r'Relative Phase Fraction $\frac{\phi_{\delta}-\phi_{\mathrm{L}}}{\phi_{\delta}+\phi_{\mathrm{L}}}$', fontsize=18)
plt.ylim([-1, 1])
plt.plot((0, tmax), (0, 0), color='black', zorder=1)

# Generate combined plots (ternary and trajectories); store individual data
fmax = 0.
datasets = []
for datdir in glob.glob('/data/tnk10/phase-field/alloy625/TKR4p158/run*'):
    if path.isdir(datdir) and len(glob.glob("{0}/c.log".format(datdir))) > 0:
        base = path.basename(datdir)
        try:
            xCr, xNb, fd, fl = np.genfromtxt("{0}/c.log".format(datdir), usecols=(2, 3, 5, 6), delimiter='\t', skip_header=1, unpack=True)
            t = dt * np.arange(0, len(fd))
            if fl[-1] > fmax:
                fmax = 1.001 * fl[-1]

            datasets.append((base, xCr[0], xNb[0], fd, fl))

            plt.figure(1)
            plt.plot(t, (fd - fl)/(fd + fl), zorder=2)
        except:
            print("Skipping {0}".format(datdir))

plt.figure(1)
plt.savefig("diagrams/TKR4p158/phases.png", dpi=400, bbox_inches='tight')
plt.close()

summary = open("TKR4p158-summary.csv", "w")
summary.write("name,x_Cr_,x_Nb_,phase\n")

for base, xCr, xNb, fd, fl in datasets:
    summary.write("{0},{1},{2},\n".format(base, xCr, xNb))
    t = dt * np.arange(0, len(fd))

    # Compute equilibrium delta fraction
    aNb = leverNb(rNb, rCr, xe_del_Nb, xe_del_Cr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr)
    aCr = leverCr(rNb, rCr, xe_del_Nb, xe_del_Cr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr)
    lAO = np.sqrt((aNb - xNb)**2 + (aCr - xCr)**2)
    lAB = np.sqrt((aNb - xe_del_Nb)**2 + (aCr - xe_del_Cr)**2)
    fd0 = lAO / lAB

    # Compute equilibrium Laves fraction
    aNb = leverNb(rNb, rCr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr, xe_del_Nb, xe_del_Cr)
    aCr = leverCr(rNb, rCr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr, xe_del_Nb, xe_del_Cr)
    lAO = np.sqrt((aNb - xNb)**2 + (aCr - xCr)**2)
    lAB = np.sqrt((aNb - xe_lav_Nb)**2 + (aCr - xe_lav_Cr)**2)
    fl0 = lAO / lAB

    # Plot coarsening trajectories (difference-over-sum data)
    plt.figure(2, figsize=(10, 7.5)) # inches
    plt.title(r'%.4fCr - %.4fNb - Ni at %.0f K' % (xCr, xNb, temp), fontsize=18)
    plt.xlabel(r'$t$ / s', fontsize=18)
    plt.ylabel(r'Relative Phase Fraction $\frac{\phi_{\delta}-\phi_{\mathrm{L}}}{\phi_{\delta}+\phi_{\mathrm{L}}}$', fontsize=18)
    plt.ylim([-1, 1])
    for ibase, ixCr, ixNb, ifd, ifl in datasets:
        it = dt * np.arange(0, len(ifd))
        plt.plot(it[::100], (ifd[::100] - ifl[::100])/(ifd[::100] + ifl[::100]), c="gray", zorder=1)
    plt.plot(t, fd / (fd + fl), c=colors[1], label=r'$\delta$', zorder=2)
    plt.plot(t, -fl / (fd + fl), c=colors[2], label="Laves", zorder=2)
    plt.plot((0, t[-1]), (0, 0), c='black', zorder=1)
    plt.plot(t, (fd - fl)/(fd + fl), c="coral", zorder=1)
    plt.legend(loc='best')
    plt.savefig("diagrams/TKR4p158/ratios/ratio_{0}.png".format(base), dpi=400, bbox_inches='tight')
    plt.close()

    # Plot phase fractions with theoretical limits
    plt.figure(3, figsize=(10, 7.5)) # inches
    plt.title(r'%.4fCr - %.4fNb - Ni at %.0f K' % (xCr, xNb, temp), fontsize=18)
    plt.xlabel(r'$t$ / s', fontsize=18)
    plt.ylabel(r'Phase Fraction', fontsize=18)
    plt.ylim([0, fmax])
    plt.plot(t, fd, c=colors[1], label=r'$f_{\delta}$', zorder=2)
    plt.plot(t, fl, c=colors[2], label=r'$f_{\mathrm{L}}$', zorder=2)
    plt.plot((0, t[-1]), (fd0, fd0), c=colors[1], ls=':', zorder=1)
    plt.plot((0, t[-1]), (fl0, fl0), c=colors[2], ls=':', zorder=1)
    plt.legend(loc='best')
    plt.savefig("diagrams/TKR4p158/phases/phase_{0}.png".format(base), dpi=400, bbox_inches='tight')
    plt.close()
summary.close()
