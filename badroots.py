
# coding: utf-8

# Overlay phase-field simulation compositions on ternary simplex

# Numerical libraries
import numpy as np

# Visualization libraries
import matplotlib.pylab as plt

from CALPHAD_energies import *

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
    # Nb-Ni edge
    xnb = 0.05*i
    Xtick.append(xnb)
    Ytick.append(-0.002)

# Plot ternary axes and labels
plt.figure(figsize=(10, 7.5)) # inches
plt.plot(XS, YS, '-k')
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
    
# Add composition pathways
gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt("data/alloy625/run1/sans_mu_smp/badroots.log", delimiter=',', unpack=True)
plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='red', s=2, zorder=1, label="bad $\gamma$")
plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='yellow', s=2, zorder=1, label="bad $\delta$")
plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='orange', s=2, zorder=1, label="bad L")
    
gam_xcr, gam_xnb, del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt("data/alloy625/run1/sans_mu_smp/gudroots.log", delimiter=',', unpack=True)
plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), color='green', s=1.25, zorder=1, label="good $\gamma$")
plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), color='teal', s=1.25, zorder=1, label="good $\delta$")
plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), color='cyan', s=1.25, zorder=1, label="good L")
    
#plt.xlim([0, 0.6])
#plt.ylim([0, rt3by2*0.6])
plt.legend(loc='best')
plt.savefig("diagrams/pathways_sans_mu.png", dpi=400, bbox_inches='tight')
plt.close()
