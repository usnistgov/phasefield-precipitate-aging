# -*- coding: utf-8 -*-

# Overlay phase-field simulation compositions on ternary phase diagram
# Usage: python analysis/TKR4p149-summary.py

import numpy as np
import matplotlib.pylab as plt
from sympy import Matrix, solve_linear_system, symbols
from sympy.abc import x, y

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

tmax = 5625 # 3750
pmax = 0.35

# Empirically determined constants for coord transformation
def coord149(n):
    tta = 0.8376
    phi = 0.0997
    psi = 0.4636
    dlx = 0.0205
    dly = 0.4018
    cr = 0.0275 + 0.05 * np.random.rand(1, n)
    nb = 0.0100 + 0.05 * np.random.rand(1, n)
    x = nb * (np.cos(tta) + np.tan(psi)) + cr * (np.sin(tta) + np.tan(phi)) + dlx
    y =-nb * (np.sin(tta) + np.tan(psi)) + cr * (np.cos(tta) + np.tan(phi)) + dly
    return (x, y)

def coord159(n):
    tta =  1.1000
    phi = -0.4000
    psi =  0.7000
    dlx =  0.0075
    dly =  0.4750
    cr = 0.0275 + 0.05 * np.random.rand(1, n)
    nb = 0.0100 + 0.05 * np.random.rand(1, n)
    x = nb * (np.cos(tta) + np.tan(psi)) + cr * (np.sin(tta) + np.tan(phi)) + dlx
    y =-nb * (np.sin(tta) + np.tan(psi)) + cr * (np.cos(tta) + np.tan(phi)) + dly
    return (x, y)

def coord160(n):
    tta =  1.1000
    phi = -0.4500
    psi =  0.8000
    dlx =  0.01
    dly =  0.50
    cr = 0.0275 + 0.05 * np.random.rand(1, n)
    nb = 0.0100 + 0.05 * np.random.rand(1, n)
    x = nb * (np.cos(tta) + np.tan(psi)) + cr * (np.sin(tta) + np.tan(phi)) + dlx
    y =-nb * (np.sin(tta) + np.tan(psi)) + cr * (np.cos(tta) + np.tan(phi)) + dly
    return (x, y)

def coordlist(n):
    dlx = 0.0275
    dly = 0.47
    scx = 0.2
    scy = 0.025
    tta = 0.925
    cr = np.random.rand(1, n)
    nb = np.random.rand(1, n)
    x = nb * scx * np.cos(tta) + cr * scy * np.sin(tta) + dlx
    y =-nb * scx * np.sin(tta) + cr * scy * np.cos(tta) + dly
    return (x, y)

def draw_bisector(A, B):
    bNb = (A * xe_del_Nb + B * xe_lav_Nb) / (A + B)
    bCr = (A * xe_del_Cr + B * xe_lav_Cr) / (A + B)
    x = [simX(xe_gam_Nb, xe_gam_Cr), simX(bNb, bCr)]
    y = [simY(xe_gam_Cr), simY(bCr)]
    return x, y

# Plot ternary axes and labels
plt.figure(0, figsize=(10, 7.5)) # inches
plt.title("Phase Coexistence at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.xticks(np.linspace(0, 1, 11))
plt.plot(XS, YS, '-k')
plt.scatter(Xtick, Ytick, color='black', s=3, zorder=5)
plt.plot(X0, Y0, color='black', zorder=5)
for a in range(len(XG)):
    plt.plot(XG[a], YG[a], ':k', linewidth=0.5, alpha=0.5, zorder=1)

# Plot generated data, colored by phase
base, xCr, xNb, phase = np.genfromtxt('analysis/TKR4p149-summary-permanent.csv', delimiter=',', dtype=None, unpack=True, skip_header=0)
dCr = np.array(xCr[phase == 'D'], dtype=float)
dNb = np.array(xNb[phase == 'D'], dtype=float)
lCr = np.array(xCr[phase == 'L'], dtype=float)
lNb = np.array(xNb[phase == 'L'], dtype=float)
plt.scatter(simX(dNb, dCr), simY(dCr), s=12, c="blue", zorder=2)
plt.scatter(simX(lNb, lCr), simY(lCr), s=12, c="orange", zorder=2)

# Plot surveyed regions using random points on [0,1)
xB, yB = draw_bisector(6., 5.)
plt.plot(xB, yB, c="green", lw=2, zorder=1)

gann = plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
plt.xlim([0.20, 0.48])
plt.ylim([0.25, 0.50])
plt.savefig("diagrams/TKR4p149/coexistence-149.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot ternary axes and labels
plt.figure(0, figsize=(10, 7.5)) # inches
plt.title("Phase Coexistence at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.xticks(np.linspace(0, 1, 11))
plt.plot(XS, YS, '-k')
plt.scatter(Xtick, Ytick, color='black', s=3, zorder=5)
plt.plot(X0, Y0, color='black', zorder=5)
for a in range(len(XG)):
    plt.plot(XG[a], YG[a], ':k', linewidth=0.5, alpha=0.5, zorder=1)
gann = plt.text(simX(0.010, 0.495), simY(0.495), r'$\gamma$', fontsize=14)
plt.xlim([0.20, 0.48])
plt.ylim([0.25, 0.50])

N = 500
coords = np.array(coordlist(N)).T.reshape((N, 2))

for rNb, rCr in coords:
    # Compute equilibrium delta fraction
    aNb = leverNb(rNb, rCr, xe_del_Nb, xe_del_Cr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr)
    aCr = leverCr(rNb, rCr, xe_del_Nb, xe_del_Cr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr)
    lAO = np.sqrt((aNb - rNb)**2 + (aCr - rCr)**2)
    lAB = np.sqrt((aNb - xe_del_Nb)**2 + (aCr - xe_del_Cr)**2)
    fd0 = lAO / lAB

    # Compute equilibrium Laves fraction
    aNb = leverNb(rNb, rCr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr, xe_del_Nb, xe_del_Cr)
    aCr = leverCr(rNb, rCr, xe_lav_Nb, xe_lav_Cr, xe_gam_Nb, xe_gam_Cr, xe_del_Nb, xe_del_Cr)
    lAO = np.sqrt((aNb - rNb)**2 + (aCr - rCr)**2)
    lAB = np.sqrt((aNb - xe_lav_Nb)**2 + (aCr - xe_lav_Cr)**2)
    fl0 = lAO / lAB

    # Collate data, colored by phase
    if fd0 >= fl0:
        plt.scatter(simX(rNb, rCr), simY(rCr), s=12, c="blue", zorder=2)
    else:
        plt.scatter(simX(rNb, rCr), simY(rCr), s=12, c="orange", zorder=2)

xB, yB = draw_bisector(6., 5.)
plt.plot(xB, yB, c="green", lw=2, zorder=1)
xB, yB = draw_bisector(1., 1.)
plt.plot(xB, yB, c="black", lw=2, zorder=1)
xB, yB = draw_bisector(5., 7.)
plt.plot(xB, yB, c="red", lw=2, zorder=1)
plt.savefig("diagrams/TKR4p149/prediction.png", dpi=300, bbox_inches='tight')
plt.close()
