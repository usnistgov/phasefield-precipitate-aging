#!/usr/bin/python
# -*- coding: utf-8 -*-

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.

# Usage: python TKR4p149.py

import glob
from os import path
import numpy as np
import matplotlib.pylab as plt
from sympy import Matrix, solve_linear_system, symbols

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

# Constants
epsilon = 1e-10 # tolerance for comparing floating-point numbers to zero
temp = 870 + 273.15 # 1143 Kelvin
fr1by2 = 1.0 / 2
rt3by2 = np.sqrt(3.0)/2
RT = 8.3144598*temp # J/mol/K
dt = 7.5e-5 * 1000
tmax = 5625 # 3750
pmax = 0.35

# Coexistence vertices
gamCr = 0.490
gamNb = 0.025
delCr = 0.015
delNb = 0.245
lavCr = 0.300
lavNb = 0.328

def simX(xnb, xcr):
    return xnb + fr1by2 * xcr

def simY(xcr):
    return rt3by2 * xcr

# Define lever rule equations
from sympy.abc import x, y
x0, y0 = symbols('x0 y0')
xb, yb = symbols('xb yb')
xc, yc = symbols('xc yc')
xd, yd = symbols('xd yd')

system = Matrix(( (y0 - yb, xb - x0, xb * (y0 - yb) + yb * (xb - x0)),
                  (yc - yd, xd - xc, xd * (yc - yd) + yd * (xd - xc)) ))
levers = solve_linear_system(system, x, y)

# triangle bounding the Gibbs simplex
XS = [0, simX(1, 0), simX(0, 1), 0]
YS = [0, simY(0), simY(1), 0]

# triangle bounding three-phase coexistence
X0 = [simX(gamNb, gamCr), simX(delNb, delCr), simX(lavNb, lavCr), simX(gamNb, gamCr)]
Y0 = [simY(gamCr), simY(delCr), simY(lavCr), simY(gamCr)]

# Tick marks along simplex edges
Xtick = []
Ytick = []
tickdens = 10
for i in range(tickdens):
    # Cr-Ni edge
    xcr = (1.0 * i) / tickdens
    xni = 1.0 - xcr
    Xtick.append(simX(-0.002, xcr))
    Ytick.append(simY(xcr))
    # Cr-Nb edge
    xcr = (1.0 * i) / tickdens
    xnb = 1.0 - xcr
    Xtick.append(simX(xnb+0.002, xcr))
    Ytick.append(simY(xcr))
    # Nb-Ni edge
    xnb = (1.0 * i) / tickdens
    Xtick.append(xnb)
    Ytick.append(-0.002)

# Triangular grid
XG = [[]]
YG = [[]]
for a in np.arange(0, 1, 0.1):
    # x1 - x2: lines of constant x2=a
    XG.append([simX(a, 0), simX(a, 1-a)])
    YG.append([simY(0),    simY(1-a)])
    # x2 - x3: lines of constant x3=a
    XG.append([simX(0, a), simX(1-a, a)])
    YG.append([simY(a),    simY(a)])
    # x1 - x3: lines of constant x1=1-a
    XG.append([simX(0, a), simX(a, 0)])
    YG.append([simY(a),    simY(0)])

# Plot ternary axes and labels
# plt.figure(0, figsize=(10, 7.5)) # inches
# plt.title("Phase Coexistence at %.0f K"%temp, fontsize=18)
# plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
# plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
# plt.xlim([0, 1])
# plt.ylim([0, rt3by2])
# plt.xticks(np.linspace(0, 1, 11))
# plt.plot(XS, YS, '-k')
# plt.scatter(Xtick, Ytick, color='black', s=3, zorder=10)
# plt.plot(X0, Y0, color='black', zorder=10)
# for a in range(len(XG)):
#     plt.plot(XG[a], YG[a], ':k', linewidth=0.5, alpha=0.5)

# Plot difference in size between delta and Laves precipitates
plt.figure(1, figsize=(10, 7.5)) # inches
plt.title("Cr-Nb-Ni at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$t$ / s', fontsize=18)
plt.ylabel(r'Relative Phase Fraction $\frac{\phi_{\delta}-\phi_{\mathrm{L}}}{\phi_{\delta}+\phi_{\mathrm{L}}}$', fontsize=18)
# plt.xlim([0, tmax])
plt.ylim([-1, 1])
plt.plot((0, tmax), (0, 0), color='black', zorder=1)

# Generate combined plots (ternary and trajectories); store individual data
fmax = 0.
datasets = []
for datdir in glob.glob('/data/tnk10/phase-field/alloy625/TKR4p149/run*'):
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

            # plt.figure(0)
            # if fd[-1] > fl[-1]:
            #     plt.scatter(simX(xNb[-1], xCr[-1]), simY(xCr[-1]), c="blue")
            # else:
            #     plt.scatter(simX(xNb[-1], xCr[-1]), simY(xCr[-1]), c="orange")
        except:
            print("Skipping {0}".format(datdir))

plt.figure(1)
plt.savefig("diagrams/TKR4p149/phases.png", dpi=400, bbox_inches='tight')
plt.close()

summary = open("TKR4p149-summary.csv", "w")
summary.write("name,x_Cr_,x_Nb_,phase\n")

envelope = ('run2', 'run3', 'run6', 'run21', 'run63')

for base, xCr, xNb, fd, fl in datasets:
    # if base in envelope:
        summary.write("{0},{1},{2},\n".format(base, xCr, xNb))
        t = dt * np.arange(0, len(fd))

        # Compute equilibrium delta fraction
        aNb = float(levers[x].subs({x0: xNb, y0: xCr, xb: delNb, yb: delCr, xc: lavNb, yc: lavCr, xd: gamNb, yd: gamCr}))
        aCr = float(levers[y].subs({x0: xNb, y0: xCr, xb: delNb, yb: delCr, xc: lavNb, yc: lavCr, xd: gamNb, yd: gamCr}))
        lAO = np.sqrt((aNb - xNb)**2 + (aCr - xCr)**2)
        lAB = np.sqrt((aNb - delNb)**2 + (aCr - delCr)**2)
        fd0 = lAO / lAB

        # Compute equilibrium Laves fraction
        aNb = float(levers[x].subs({x0: xNb, y0: xCr, xb: lavNb, yb: lavCr, xc: gamNb, yc: gamCr, xd: delNb, yd: delCr}))
        aCr = float(levers[y].subs({x0: xNb, y0: xCr, xb: lavNb, yb: lavCr, xc: gamNb, yc: gamCr, xd: delNb, yd: delCr}))
        lAO = np.sqrt((aNb - xNb)**2 + (aCr - xCr)**2)
        lAB = np.sqrt((aNb - lavNb)**2 + (aCr - lavCr)**2)
        fl0 = lAO / lAB

        # Plot coarsening trajectories (difference-over-sum data)
        plt.figure(2, figsize=(10, 7.5)) # inches
        plt.title(r'%.4fCr - %.4fNb - Ni at %.0f K' % (xCr, xNb, temp), fontsize=18)
        plt.xlabel(r'$t$ / s', fontsize=18)
        plt.ylabel(r'Relative Phase Fraction $\frac{\phi_{\delta}-\phi_{\mathrm{L}}}{\phi_{\delta}+\phi_{\mathrm{L}}}$', fontsize=18)
        # plt.xlim([0, tmax])
        plt.ylim([-1, 1])
        for ibase, ixCr, ixNb, ifd, ifl in datasets:
            it = dt * np.arange(0, len(ifd))
            plt.plot(it[::100], (ifd[::100] - ifl[::100])/(ifd[::100] + ifl[::100]), c="gray", zorder=1)
        plt.plot(t, fd / (fd + fl), c=colors[1], label=r'$\delta$', zorder=2)
        plt.plot(t, -fl / (fd + fl), c=colors[2], label="Laves", zorder=2)
        plt.plot((0, t[-1]), (0, 0), c='black', zorder=1)
        plt.plot(t, (fd - fl)/(fd + fl), c="coral", zorder=1)
        plt.legend(loc='best')
        plt.savefig("diagrams/TKR4p149/ratio_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()

        # Plot phase fractions with theoretical limits
        plt.figure(3, figsize=(10, 7.5)) # inches
        plt.title(r'%.4fCr - %.4fNb - Ni at %.0f K' % (xCr, xNb, temp), fontsize=18)
        plt.xlabel(r'$t$ / s', fontsize=18)
        plt.ylabel(r'Phase Fraction', fontsize=18)
        # plt.xlim([0, tmax])
        plt.ylim([0, fmax])
        plt.plot(t, fd, c=colors[1], label=r'$f_{\delta}$', zorder=2)
        plt.plot(t, fl, c=colors[2], label=r'$f_{\mathrm{L}}$', zorder=2)
        plt.plot((0, t[-1]), (fd0, fd0), c=colors[1], ls=':', zorder=1)
        plt.plot((0, t[-1]), (fl0, fl0), c=colors[2], ls=':', zorder=1)
        plt.legend(loc='best')
        plt.savefig("diagrams/TKR4p149/phase_{0}.png".format(base), dpi=400, bbox_inches='tight')
        plt.close()
summary.close()
