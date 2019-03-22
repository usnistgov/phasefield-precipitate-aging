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
from pyCinterface import *

pltsize = 10
density = 1000
ncontour = 200
fceil = 3e9
colorSet = plt.cm.gray(np.linspace(0, 1, 11))
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'center'}

def labelAxes(xlabel, ylabel, n):
  plt.axis('off')
  plt.gca().set_aspect('equal', adjustable='box')
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, simY(1.05)])

  def plot_ticks(start, stop, tick, angle, n):
    plt.text(0.5, -0.075, xlabel, fontsize=18, color=colorSet[1])
    plt.text(simX(-0.11, 0.55), simY(0.55), ylabel, rotation=60, fontsize=18, color=colorSet[1])

    # from https://stackoverflow.com/a/30975434/5377275
    dx = 3 * tick[0]
    dy = 3 * tick[1]
    r = np.linspace(0, 1, n+1)
    x = start[0] * (1 - r) + stop[0] * r
    y = start[1] * (1 - r) + stop[1] * r
    if angle >= 0:
      for i in range(len(x)):
        plt.text(x[i] + dx , y[i] + dy, "{0:.1f}".format(r[i]), rotation=angle, color=colorSet[1], **alignment)
    else:
      midx = 0.5 * (start[0] + stop[0])
      midy = 0.5 * (start[1] + stop[1])
      plt.text(midx + dx, midy + dy, "0.5", rotation=angle, color=colorSet[1], **alignment)
    x = np.vstack((x, x + tick[0]))
    y = np.vstack((y, y + tick[1]))
    plt.plot(x, y, 'k', lw=1, color=colorSet[1])

  # Spatial considerations
  tick_size = 0.075
  left = np.r_[0, 0]
  right = np.r_[1, 0]
  top = np.r_[simX(0, 1), simY(1)]

  # define vectors for ticks
  bottom_tick = tick_size * np.r_[0, -1] / n
  right_tick = sqrt(3) * tick_size * np.r_[1,0.5] * (top - left) / n
  left_tick = sqrt(3) * tick_size * np.r_[1,0.5] * (top - right) / n

  XS = [0, simX(1,0), simX(0,1), 0]
  YS = [0, simY(0),   simY(1),   0]
  plt.plot(XS, YS, '-k', zorder=2, color=colorSet[1])
  plot_ticks(left, right, bottom_tick, 0, n)
  plot_ticks(right, top, right_tick, -60, n)
  plot_ticks(left, top, left_tick, 60, n)

fig = plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18, color=colorSet[1])
labelAxes(r'$x_{\mathrm{Nb}}$', r'$x_{\mathrm{Cr}}$', 10)

plt.text(0.29, 0.22, '$\gamma$',  color=colorSet[9], fontsize=18, **alignment)
plt.text(0.30, 0.16, '$\delta$',  color=colorSet[9], fontsize=18, **alignment)
plt.text(0.33, 0.21, '$\lambda$', color=colorSet[9], fontsize=18, **alignment)

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

levels = np.linspace(0, fceil, ncontour)

plt.tricontourf(x, y, z, levels, cmap=plt.cm.get_cmap('binary'))

plt.savefig("ternary-landscape.png", dpi=600, bbox_inches="tight",
            facecolor=colorSet[0], edgecolor=None)
