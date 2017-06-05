#!/usr/bin/python
# coding: utf-8

# Plot compositions in ternary simplex

# Numerical libraries
import numpy as np
from os import path, stat
from sys import argv

# Visualization libraries
import matplotlib.pylab as plt

def simX(x2, x3):
    return x2 + fr1by2 * x3

def simY(x3):
    return rt3by2 * x3

fr1by2 = 1.0/2
rt3by2 = np.sqrt(3)/2

# triangle bounding the Gibbs simplex
XS = [0.0, simX(1,0), simX(0,1), 0.0]
YS = [0.0, simY(0),   simY(1),   0.0]

# triangle bounding three-phase coexistence
X0 = [simX(0.025, 0.490), simX(0.245, 0.02), simX(0.283, 0.30)]
Y0 = [simY(0.490),simY(0.02),simY(0.30)]

# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    xni = 1.0 - xcr
    Xtick.append(simX(-0.002, xcr))
    Ytick.append(simY(xcr))
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.0 - xcr
    Xtick.append(simX(xnb+0.002, xcr))
    Ytick.append(simY(xcr))
    # Nb-Ni edge
    xnb = 0.05*i
    Xtick.append(xnb)
    Ytick.append(-0.002)

# Plot ternary axes and labels
plt.figure(figsize=(10, 7.5)) # inches
plt.plot(XS, YS, '-k')
plt.title("Cr-Nb-Ni Equilibrium at 1143 K", fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3, zorder=10)
plt.scatter(X0, Y0, color='black', s=3, zorder=10)
 
# Plot compositions
gam_del_eqm = "gamma_delta_eqm.txt"
gam_lav_eqm = "gamma_laves_eqm.txt"
del_lav_eqm = "delta_laves_eqm.txt"

if stat(gam_del_eqm).st_size > 0:
    gam_xcr, gam_xnb, del_xcr, del_xnb = np.loadtxt(gam_del_eqm, delimiter='\t', unpack=True)
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), c='red', s=1)
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), c='green', s=1)

if stat(gam_lav_eqm).st_size > 0:
    gam_xcr, gam_xnb, lav_xcr, lav_xnb = np.loadtxt(gam_lav_eqm, delimiter='\t', unpack=True)
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), c='orange', s=1)
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), c='blue', s=1)

if stat(del_lav_eqm).st_size > 0:
    del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(del_lav_eqm, delimiter='\t', unpack=True)
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), c='teal', s=1)
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), c='cyan', s=1)

plt.xlim([0, 1])
plt.ylim([0, rt3by2])
plt.savefig("equilibrium_phase_diagram.png", dpi=400, bbox_inches='tight')
plt.close()
