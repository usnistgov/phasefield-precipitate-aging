#!/usr/bin/python
# coding: utf-8

# Plot compositions in ternary simplex

# Numerical libraries
import numpy as np
from os import stat

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
plt.title("Cr-Nb-Ni Fictitious Compositions", fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
 
# Plot compositions given to rootsolver
thebad = "data/alloy625/run1/sans_mu_smp/badroots.log"
if stat(thebad).st_size > 0:
    gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(thebad, delimiter=',', unpack=True)
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='red', s=2, zorder=1, label="bad $\gamma$")
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='yellow', s=2, zorder=1, label="bad $\delta$")
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='orange', s=2, zorder=1, label="bad L")

thegud = "data/alloy625/run1/sans_mu_smp/gudroots.log" 
if stat(thegud).st_size > 0:
    gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(thegud, delimiter=',', unpack=True)
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='green', s=1.25, zorder=1, label="good $\gamma$")
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='teal', s=1.25, zorder=1, label="good $\delta$")
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='cyan', s=1.25, zorder=1, label="good L")

plt.legend(loc='best')
plt.savefig("diagrams/pathways_sans_mu.png", dpi=400, bbox_inches='tight')
plt.close()
