# -*- coding: utf-8 -*-

# Generate ternary phase diagram
# Usage: python ternary-landscape.py

from math import ceil, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import fsolve

from constants import *
from pyCinterface import *

pltsize = 10
density = 1000
ncontour = 200
fceil = 3e9
grScl = 21
dark = grScl / 10
light = grScl - 2
grays = plt.cm.gray(np.linspace(0, 1, grScl))
alignment = {"horizontalalignment": "center", "verticalalignment": "center"}


def labelAxes(xlabel, ylabel, n):
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, simY(1.05)])

    def plot_ticks(start, stop, tick, angle, n):
        plt.text(0.5, -0.075, xlabel, fontsize=18, color=grays[dark])
        plt.text(
            simX(-0.11, 0.55),
            simY(0.55),
            ylabel,
            rotation=60,
            fontsize=18,
            color=grays[dark],
        )

        # from https://stackoverflow.com/a/30975434/5377275
        dx = 3 * tick[0]
        dy = 3 * tick[1]
        r = np.linspace(0, 1, n + 1)
        x = start[0] * (1 - r) + stop[0] * r
        y = start[1] * (1 - r) + stop[1] * r
        if angle >= 0:
            for i in range(len(x)):
                plt.text(
                    x[i] + dx,
                    y[i] + dy,
                    "{0:.1f}".format(r[i]),
                    rotation=angle,
                    color=grays[dark],
                    **alignment
                )
        else:
            midx = 0.5 * (start[0] + stop[0])
            midy = 0.5 * (start[1] + stop[1])
            plt.text(
                midx + dx,
                midy + dy,
                "0.5",
                rotation=angle,
                color=grays[dark],
                **alignment
            )
        x = np.vstack((x, x + tick[0]))
        y = np.vstack((y, y + tick[1]))
        plt.plot(x, y, "k", lw=1, color=grays[dark])

    # Spatial considerations
    tick_size = 0.075
    left = np.r_[0, 0]
    right = np.r_[1, 0]
    top = np.r_[simX(0, 1), simY(1)]

    # define vectors for ticks
    bottom_tick = tick_size * np.r_[0, -1] / n
    right_tick = sqrt(3) * tick_size * np.r_[1, 0.5] * (top - left) / n
    left_tick = sqrt(3) * tick_size * np.r_[1, 0.5] * (top - right) / n

    XS = [0, simX(1, 0), simX(0, 1), 0]
    YS = [0, simY(0), simY(1), 0]
    plt.plot(XS, YS, "-k", zorder=2, color=grays[dark])
    plot_ticks(left, right, bottom_tick, 0, n)
    plot_ticks(right, top, right_tick, -60, n)
    plot_ticks(left, top, left_tick, 60, n)


fig = plt.figure(figsize=(pltsize, rt3by2 * pltsize))
plt.title("Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18, color=grays[dark])
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)

plt.text(
    simX(0.050, 0.450),
    simY(0.450),
    "$\gamma$",
    color="black",
    alpha=0.1,
    fontsize=20,
    **alignment
)
plt.text(
    simX(0.235, 0.050),
    simY(0.050),
    "$\delta$",
    color="black",
    alpha=0.1,
    fontsize=20,
    **alignment
)
plt.text(
    simX(0.280, 0.365),
    simY(0.365),
    "$\lambda$",
    color="black",
    alpha=0.1,
    fontsize=20,
    **alignment
)

x = []
y = []
z = []

for xNbTest in np.linspace(0, 1, density):
    for xCrTest in np.linspace(0, 1 - xNbTest, max(1, ceil((1 - xNbTest) * density))):
        fGam = g_gam(xCrTest, xNbTest)
        fDel = g_del(xCrTest, xNbTest)
        fLav = g_lav(xCrTest, xNbTest)

        minima = np.asarray([fGam, fDel, fLav])
        minidx = np.argmin(minima)

        x.append(simX(xNbTest, xCrTest))
        y.append(simY(xCrTest))

        if minidx == 0:
            z.append(fGam)
        elif minidx == 1:
            z.append(fDel)
        elif minidx == 2:
            z.append(fLav)

fmin = min(z)
fmax = max(z)

x = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)

z[z > fceil] = fceil

levels = np.linspace(0, fceil, ncontour)

plt.tricontourf(x, y, z, levels, cmap=plt.cm.get_cmap("binary"))

npz = np.load("tie-lines.npz")
for a, b in npz["arr_0"]:  # sAB
    plt.scatter(a, b, color="black", alpha=0.010, s=0.5)
for a, c in npz["arr_1"]:  # sAC
    plt.scatter(a, c, color="black", alpha=0.010, s=0.5)
for b, c in npz["arr_2"]:  # sBC
    plt.scatter(b, c, color="black", alpha=0.010, s=0.5)
for a, b in npz["arr_3"][::10]:  # sBA
    plt.scatter(a, b, color="black", alpha=0.010, s=0.5)
for a, c in npz["arr_4"][::10]:  # sCA
    plt.scatter(a, c, color="black", alpha=0.010, s=0.5)
for b, c in npz["arr_5"]:  # sCB
    plt.scatter(b, c, color="black", alpha=0.010, s=0.5)

npz = np.load("metastable-lines.npz")
for a, b in npz["arr_0"]:  # mAB
    plt.scatter(a, b, color="black", alpha=0.005, s=0.5)
for a, c in npz["arr_1"]:  # mAC
    plt.scatter(a, c, color="black", alpha=0.005, s=0.5)
for b, c in npz["arr_2"]:  # mBC
    plt.scatter(b, c, color="black", alpha=0.005, s=0.5)
for a, b in npz["arr_3"][::10]:  # mBA
    plt.scatter(a, b, color="black", alpha=0.005, s=0.5)
for a, c in npz["arr_4"][::10]:  # mCA
    plt.scatter(a, c, color="black", alpha=0.005, s=0.5)
for b, c in npz["arr_5"]:  # mCB
    plt.scatter(b, c, color="black", alpha=0.005, s=0.5)

plt.savefig(
    "ternary-landscape.png",
    dpi=400,
    bbox_inches="tight",
    facecolor="black",
    edgecolor=None,
)
