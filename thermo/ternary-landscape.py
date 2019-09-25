# -*- coding: utf-8 -*-

# Generate ternary phase diagram
# Usage: python ternary-landscape.py

from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from pyCinterface import *

pltsize = 10
density = 1000
ncontour = 200
grScl = 21
dark = int(grScl / 10)
light = int(grScl - 4)
grays = plt.cm.gray(np.linspace(0, 1, grScl))
alignment = {"horizontalalignment": "center", "verticalalignment": "center"}

xlGCr = 0.51
xlGNb = 0.02
xlGNi = 1 - xlGCr - xlGNb
xlDCr = 0.01
xlDNb = 0.25
xlDNi = 1 - xlDCr - xlDNb
xlLCr = 0.35
xlLNb = 0.27
xlLNi = 1 - xlLCr - xlLNb

def labelAxes(xlabel, ylabel, n):
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, simY(1.05)])

    def plot_ticks(start, stop, tick, angle, n):
        plt.text(0.5, -0.075, xlabel, fontsize=18, color=grays[light])
        plt.text(
            simX(-0.11, 0.55),
            simY(0.55),
            ylabel,
            rotation=60,
            fontsize=18,
            color=grays[light],
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
                    color=grays[light],
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
                color=grays[light],
                **alignment
            )
        x = np.vstack((x, x + tick[0]))
        y = np.vstack((y, y + tick[1]))
        plt.plot(x, y, "k", lw=1, color=grays[light])

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
    plt.plot(XS, YS, "-k", zorder=2, color=grays[light])
    plot_ticks(left, right, bottom_tick, 0, n)
    plot_ticks(right, top, right_tick, -60, n)
    plot_ticks(left, top, left_tick, 60, n)

x = []
y = []
zc = []
zp = []

for xNbTest in np.linspace(0, 1, density):
    for xCrTest in np.linspace(0, 1 - xNbTest, max(1, ceil((1 - xNbTest) * density))):
        x.append(simX(xNbTest, xCrTest))
        y.append(simY(xCrTest))

        cGam = CALPHAD_gam(xCrTest, xNbTest)
        cDel = 0 if xNbTest > 0.25 else CALPHAD_del(xCrTest, xNbTest)
        cLav = 0 if xNbTest > 0.3333 else CALPHAD_lav(xCrTest, xNbTest)

        minima = np.asarray([cGam, cDel, cLav])
        minidx = np.argmin(minima)

        if minidx == 0:
            zc.append(cGam)
        elif minidx == 1:
            zc.append(cDel)
        elif minidx == 2:
            zc.append(cLav)

        pGam = g_gam(xCrTest, xNbTest)
        pDel = g_del(xCrTest, xNbTest)
        pLav = g_lav(xCrTest, xNbTest)

        minima = np.asarray([pGam, pDel, pLav])
        minidx = np.argmin(minima)

        if minidx == 0:
            zp.append(pGam)
        elif minidx == 1:
            zp.append(pDel)
        elif minidx == 2:
            zp.append(pLav)

x = np.asarray(x)
y = np.asarray(y)

zc = np.asarray(zc)
fcmin = min(zc)
fcmax = max(zc)
fcran = fcmax - fcmin

print("CALPHAD energies span [{0:10.3e}, {1:10.3e}]; range is {2:10.3e}".format(fcmin, fcmax, fcran))

zp = np.asarray(zp)
fpmin = min(zp)
zp[zp > fpmin + fcran] = fpmin + fcran # Danger: data manipulation!
fpmax = max(zp)
fpran = fpmax - fpmin

print("Parabol energies span [{0:10.3e}, {1:10.3e}]; range is {2:10.3e}".format(fpmin, fpmax, fpran))


fig = plt.figure(figsize=(2 * pltsize, rt3by2 * pltsize))

plt.subplot(1, 2, 1)
plt.title("CALPHAD", fontsize=16, color=grays[light])
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)

levels = np.linspace(fcmin, fcmax, ncontour)
plt.tricontourf(x, y, zc, levels, cmap=plt.cm.get_cmap("binary"))

plt.subplot(1, 2, 2)
plt.title("Paraboloid", fontsize=16, color=grays[light])
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)

levels = np.linspace(fpmin, fpmax, ncontour)
plt.tricontourf(x, y, zp, levels, cmap=plt.cm.get_cmap("binary"))

plt.text(-0.25, 0.95, "Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18, color=grays[light])

fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig(
    "ternary-landscape.png",
    dpi=400,
    bbox_inches="tight",
    facecolor="black",
    edgecolor=None,
)

plt.close()

plt.figure(figsize=(pltsize, rt3by2 * pltsize))
plt.title("CALPHAD Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18, color=grays[light])
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)
levels = np.linspace(fcmin, fcmax, ncontour)
plt.tricontourf(x, y, zc, levels, cmap=plt.cm.get_cmap("binary"))
plt.text(simX(xlGNb, xlGCr), simY(xlGCr), "$\gamma$",
         color=grays[light], fontsize=16, zorder=2, **alignment)
plt.text(xlDNb, xlDCr, "$\delta$",
         color=grays[dark], fontsize=16, zorder=2, **alignment)
plt.text(simX(xlLNb, xlLCr), simY(xlLCr), "$\lambda$",
    color=grays[dark], fontsize=16, zorder=2, **alignment)
plt.savefig("calphad-landscape.png", dpi=400, bbox_inches="tight", facecolor="black", edgecolor=None)
plt.close()

plt.figure(figsize=(pltsize, rt3by2 * pltsize))
plt.title("Paraboloid Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18, color=grays[light])
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)
levels = np.linspace(fpmin, fpmax, ncontour)
plt.tricontourf(x, y, zp, levels, cmap=plt.cm.get_cmap("binary"))
plt.text(simX(xlGNb, xlGCr), simY(xlGCr), "$\gamma$",
         color=grays[dark], fontsize=16, zorder=2, **alignment)
plt.text(xlDNb, xlDCr, "$\delta$",
         color=grays[dark], fontsize=16, zorder=2, **alignment)
plt.text(simX(xlLNb, xlLCr), simY(xlLCr), "$\lambda$",
    color=grays[dark], fontsize=16, zorder=2, **alignment)
plt.savefig("paraboloid-landscape.png", dpi=400, bbox_inches="tight", facecolor="black", edgecolor=None)
plt.close()
