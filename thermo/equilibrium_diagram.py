# -*- coding: utf-8 -*-

# Plot compositions in ternary simplex
# Usage: python equilibrium_diagram.py

# Numerical libraries
import numpy as np
from os import path, stat
from sys import argv
import matplotlib.pylab as plt
from constants import *

# Plot ternary axes and labels
plt.figure(figsize=(10, 7.5))  # inches
plt.plot(XS, YS, "-k")
plt.title("Cr-Nb-Ni Equilibrium at 1143 K", fontsize=18)
plt.xlabel(r"$x_\mathrm{Nb}$", fontsize=24)
plt.ylabel(r"$x_\mathrm{Cr}$", fontsize=24)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color="black", s=3, zorder=10)
plt.scatter(X0, Y0, color="black", s=3, zorder=10)

# Plot compositions
gam_del_eqm = "gamma_delta_eqm.txt"
gam_lav_eqm = "gamma_laves_eqm.txt"
del_lav_eqm = "delta_laves_eqm.txt"

if stat(gam_del_eqm).st_size > 0:
    gam_xcr, gam_xnb, del_xcr, del_xnb = np.loadtxt(
        gam_del_eqm, delimiter="\t", unpack=True
    )
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), c="red", s=1)
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), c="green", s=1)

if stat(gam_lav_eqm).st_size > 0:
    gam_xcr, gam_xnb, lav_xcr, lav_xnb = np.loadtxt(
        gam_lav_eqm, delimiter="\t", unpack=True
    )
    plt.scatter(simX(gam_xnb, gam_xcr), simY(gam_xcr), c="orange", s=1)
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), c="blue", s=1)

if stat(del_lav_eqm).st_size > 0:
    del_xcr, del_xnb, lav_xcr, lav_xnb = np.loadtxt(
        del_lav_eqm, delimiter="\t", unpack=True
    )
    plt.scatter(simX(del_xnb, del_xcr), simY(del_xcr), c="teal", s=1)
    plt.scatter(simX(lav_xnb, lav_xcr), simY(lav_xcr), c="cyan", s=1)

plt.xlim([0, 1])
plt.ylim([0, rt3by2])
plt.savefig("../diagrams/equilibrium_phase_diagram.png", dpi=400, bbox_inches="tight")
plt.close()
