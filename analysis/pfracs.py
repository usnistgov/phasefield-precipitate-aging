# -*- coding: utf-8 -*-

# Overlay phase-field simulation compositions on ternary phase diagram
# Before executing this script, run the mmsp2comp utility
# for each checkpoint file in the directories of interest.
# Usage: python analysis/pfrac.py data/alloy625/run2/TKR4p119/run2*

# Numerical libraries
import numpy as np
from math import floor, sqrt

# Runtime / parallel libraries
from os import path, stat
from sys import argv

import glob, time
from itertools import chain
from multiprocessing import Pool

# Visualization libraries
import matplotlib.pylab as plt

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CALPHAD_energies import *

labels = [r"$\gamma$", r"$\delta$", "Laves"]
colors = ["red", "green", "blue"]

plt.figure(0, figsize=(10, 7.5))  # inches
plt.title("Cr-Nb-Ni at %.0f K" % temp, fontsize=18)
plt.xlabel(r"$t$", fontsize=18)
plt.ylabel(r"Phase fraction $\phi$", fontsize=18)
plt.xlim([0, 50e6])
plt.ylim([0, 0.2])

for j in range(1, len(argv)):
    datdir = argv[j]
    if path.isdir(datdir) and len(glob.glob("{0}/*.xy".format(datdir))) > 0:
        base = path.basename(datdir)
        # Plot phase evolution trajectories
        t, fd, fl, fg = np.loadtxt(
            "{0}/phasefrac.csv".format(datdir), delimiter=",", skiprows=1, unpack=True
        )
        plt.figure(1, figsize=(10, 7.5))  # inches
        plt.title("Cr-Nb-Ni at %.0f K" % temp, fontsize=18)
        plt.xlabel(r"$t$", fontsize=18)
        plt.ylabel(r"Phase fraction $\phi$", fontsize=18)
        # plt.scatter(t, fg, c=colors[0], label="$\gamma$")
        plt.scatter(t, fd, c=colors[1], label="$\delta$")
        plt.plot(t, fd, c=colors[1])
        plt.scatter(t, fl, c=colors[2], label="Laves")
        plt.plot(t, fl, c=colors[2])
        plt.xlim([0, 50e6])
        plt.ylim([0, 0.15])
        plt.legend(loc="best")
        plt.savefig(
            "diagrams/phasefrac_{0}.png".format(base), dpi=400, bbox_inches="tight"
        )
        plt.close()
        plt.figure(0)
        # plt.scatter(t, fg, c=colors[0], label="$\gamma$")
        plt.scatter(t, fd, c=colors[1])
        plt.plot(t, fd, c=colors[1])
        plt.scatter(t, fl, c=colors[2])
        plt.plot(t, fl, c=colors[2])
    else:
        print(
            "Invalid argument: {0} is not a directory, or contains no usable data.".format(
                datdir
            )
        )
        print("Usage: {0} path/to/data".format(argv[0]))

plt.savefig("diagrams/phasefracs.png", dpi=400, bbox_inches="tight")
plt.figure(0)
plt.close()
