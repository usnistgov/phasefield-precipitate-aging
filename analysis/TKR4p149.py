# -*- coding: utf-8 -*-

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.
# Usage: python analysis/TKR4p149.py

import glob
from os import path
import numpy as np
import matplotlib.pylab as plt
from sympy import Matrix, solve_linear_system, symbols
from tqdm import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

tmax = 5625 # 3750
pmax = 0.35
dt = 7.5e-5 * 1000
area = 320. * 192. * (5e-9)**2

# Plot difference in size between delta and Laves precipitates
plt.figure(1, figsize=(10, 7.5)) # inches
plt.title("Cr-Nb-Ni at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$t$ / s', fontsize=18)
plt.ylabel(r'Relative Phase Fraction $\frac{\phi_{\delta}-\phi_{\mathrm{L}}}{\phi_{\delta}+\phi_{\mathrm{L}}}$', fontsize=18)
plt.ylim([-1, 1])
plt.plot((0, tmax), (0, 0), color='black', zorder=1)

# Generate combined plots (ternary and trajectories); store individual data
datasets = []
fmax = 0.
for datdir in tqdm(glob.glob('/data/tnk10/phase-field/alloy625/TKR4p149/run*/')):
    if path.isdir(datdir) and len(glob.glob("{0}/c.log".format(datdir))) > 0:
        base = path.basename(datdir)
        try:
            xCr, xNb, fd, fl = np.genfromtxt("{0}/c.log".format(datdir), usecols=(2, 3, 5, 6), delimiter='\t', skip_header=1, unpack=True)
            t = dt * np.arange(0, len(fd))
            datasets.append((base, xCr[0], xNb[0], fd, fl))
            plt.figure(1)
            plt.plot(t, (fd - fl)/(fd + fl), zorder=2)

            # Compute maximum plot value
            r_delta = np.sqrt(area * fd[0] / np.pi)
            P_delta = 2. * s_delta / r_delta
            r_laves = np.sqrt(area * fl[0] / np.pi)
            P_laves = 2. * s_laves / r_laves
            dx_gam_Cr = xe_gam_Cr + DXAB(P_delta, P_laves)
            dx_gam_Nb = xe_gam_Nb + DXAC(P_delta, P_laves)
            dx_del_Cr = xe_del_Cr + DXBB(P_delta, P_laves)
            dx_del_Nb = xe_del_Nb + DXBC(P_delta, P_laves)
            dx_lav_Cr = xe_lav_Cr + DXGB(P_delta, P_laves)
            dx_lav_Nb = xe_lav_Nb + DXGC(P_delta, P_laves)
            aCr = leverCr(xNb[0], xCr[0], dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr, dx_del_Nb, dx_del_Cr)
            aNb = leverNb(xNb[0], xCr[0], dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr, dx_del_Nb, dx_del_Cr)
            fl0 = np.sqrt(((aNb - xNb[0])**2 + (aCr - xCr[0])**2) / ((aNb - dx_lav_Nb)**2 + (aCr - dx_lav_Cr)**2))
            if fl0 > fmax:
                fmax = 1.002 * fl0
        except:
            print("Skipping {0}".format(datdir))

plt.figure(1)
plt.savefig("diagrams/TKR4p149/phases.png", dpi=400, bbox_inches='tight')
plt.close()

summary = open("analysis/TKR4p149-summary.csv", "w")
summary.write("name,x_Cr_,x_Nb_,phase\n")

for base, xCr, xNb, fd, fl in tqdm(datasets):
    summary.write("{0},{1},{2},\n".format(base, xCr, xNb))
    t = dt * np.arange(0, len(fd))
    r_delta = np.sqrt(area * fd / np.pi)
    r_laves = np.sqrt(area * fl / np.pi)
    P_delta = 2. * s_delta / r_delta
    P_laves = 2. * s_laves / r_laves

    # Compute curvature offsets
    dx_gam_Cr = xe_gam_Cr + DXAB(P_delta, P_laves)
    dx_gam_Nb = xe_gam_Nb + DXAC(P_delta, P_laves)
    dx_del_Cr = xe_del_Cr + DXBB(P_delta, P_laves)
    dx_del_Nb = xe_del_Nb + DXBC(P_delta, P_laves)
    dx_lav_Cr = xe_lav_Cr + DXGB(P_delta, P_laves)
    dx_lav_Nb = xe_lav_Nb + DXGC(P_delta, P_laves)

    # Compute equilibrium delta fraction
    aCr = leverCr(xNb, xCr, dx_del_Nb, dx_del_Cr, dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr)
    aNb = leverNb(xNb, xCr, dx_del_Nb, dx_del_Cr, dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr)
    fd0 = np.sqrt(((aNb - xNb)**2 + (aCr - xCr)**2) / ((aNb - dx_del_Nb)**2 + (aCr - dx_del_Cr)**2))

    # Compute equilibrium Laves fraction
    aCr = leverCr(xNb, xCr, dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr, dx_del_Nb, dx_del_Cr)
    aNb = leverNb(xNb, xCr, dx_lav_Nb, dx_lav_Cr, dx_gam_Nb, dx_gam_Cr, dx_del_Nb, dx_del_Cr)
    fl0 = np.sqrt(((aNb - xNb)**2 + (aCr - xCr)**2) / ((aNb - dx_lav_Nb)**2 + (aCr - dx_lav_Cr)**2))

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
    plt.savefig("diagrams/TKR4p149/ratios/ratio_{0}.png".format(base), dpi=400, bbox_inches='tight')
    plt.close()

    # Plot phase fractions with theoretical limits
    plt.figure(3, figsize=(10, 7.5)) # inches
    plt.title(r'%.4fCr - %.4fNb - Ni at %.0f K' % (xCr, xNb, temp), fontsize=18)
    plt.xlabel(r'$t$ / s', fontsize=18)
    plt.ylabel(r'Phase Fraction', fontsize=18)
    plt.ylim([0, fmax])
    plt.plot(t, fd, c=colors[1], label=r'$f_{\delta}$', zorder=2)
    plt.plot(t, fl, c=colors[2], label=r'$f_{\mathrm{L}}$', zorder=2)
    plt.plot(t, fd0, c=colors[1], ls=":", zorder=1)
    plt.plot(t, fl0, c=colors[2], ls=":", zorder=1)
    plt.legend(loc='best')
    plt.savefig("diagrams/TKR4p149/phases/phase_{0}.png".format(base), dpi=400, bbox_inches='tight')
    plt.close()

summary.close()
