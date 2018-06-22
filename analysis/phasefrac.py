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
# Usage: python analysis/pathways.py data/alloy625/run2/TKR4p119/run2*

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
t, fd, fl, fg = np.loadtxt('/data/tnk10/phase-field/alloy625/run2/TKR4p119/run21/phasefrac.csv', delimiter=',', skiprows=1, unpack=True)
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
