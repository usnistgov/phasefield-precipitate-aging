# -*- coding: utf-8 -*-

# Generate ternary phase diagram
# Usage: python ternary-landscape.py

from math import ceil, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm

from constants import *
from parabola625 import g_gam, g_del, g_lav


pltsize = 20
density = 2000
ncontour = 200
fceil = 3e9

plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.axis('off')

x = []
y = []
z = []

for xNbTest in tqdm(np.linspace(0, 1, density)):
  for xCrTest in np.linspace(0, 1 - xNbTest, max(1, ceil((1 - xNbTest) * density))):
    fGam = g_gam(xCrTest, xNbTest)
    fDel = g_del(xCrTest, xNbTest)
    fLav = g_lav(xCrTest, xNbTest)

    minima = np.asarray([fGam, fDel, fLav])
    minidx = np.argmin(minima)

    x.append(simX(xNbTest, xCrTest))
    y.append(simY(xCrTest))

    if (minidx == 0):
      z.append(fGam)
    elif (minidx == 1):
      z.append(fDel)
    elif (minidx == 2):
      z.append(fLav)

fmin = min(z)
fmax = max(z)
print "Raw data spans [{0:2.2e}, {1:2.2e}]. Truncating to {2:2.2e}.".format(fmin, fmax, fceil)

x = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)

z[z > fceil] = fceil

# levels = np.logspace(np.log2(fmin), np.log2(fmax), num=ncontour, base=2)
levels = np.linspace(0, fceil, ncontour)

# plt.tricontourf(x, y, z, levels, cmap=plt.cm.get_cmap('gray'), norm=LogNorm())
plt.tricontourf(x, y, z, levels, cmap=plt.cm.get_cmap('binary'))

plt.savefig("ternary-landscape.png", dpi=600, bbox_inches="tight", transparent=True)
