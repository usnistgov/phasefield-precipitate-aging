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

temp = 273.15 + 870

labels = [r'$\gamma$', r'$\delta$', 'Laves']
colors = ['red', 'green', 'blue']

# Plot phase evolution trajectories
t, fd, fl, fg = np.loadtxt("data/alloy625/run2/TKR4p119/run21/phasefrac.csv", delimiter=',', skiprows=1, unpack=True)
plt.figure(figsize=(10, 7.5)) # inches
plt.title("Cr-Nb-Ni at %.0f K"%temp, fontsize=18)
plt.xlabel(r'$t$', fontsize=18)
plt.ylabel(r'Phase fraction $\phi$', fontsize=18)
plt.scatter(t, fg, c=colors[0], label="$\gamma$")
plt.scatter(t, fd, c=colors[1], label="$\delta$")
plt.scatter(t, fl, c=colors[2], label="Laves")
plt.xlim([0, 50e6])
plt.ylim([0, 1.5e-12])
plt.legend(loc='best')
plt.savefig("diagrams/phasefrac_run21.png", dpi=400, bbox_inches='tight')
plt.close()
