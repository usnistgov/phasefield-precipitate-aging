# -*- coding: utf-8 -*-

# Generate phase diagrams, with compositions in mole fractions
# Ref: TKR5p234
# Usage: python phasediagram.py

from math import ceil, fabs, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import warnings

from pyCinterface import *
from constants import *

density = 201

colors = ["red", "green", "blue", "gray"]
salmon = "#fa8072"
rust = "#b7410e"

alignment = {"horizontalalignment": "center", "verticalalignment": "center"}
warnings.filterwarnings("ignore", "The iteration is not making good progress")
warnings.filterwarnings("ignore", "The number of calls to function has reached maxfev = 500.")

# Helper functions to convert compositions into (x,y) coordinates
def simX(x1, x2):
    return x1 + 0.5 * x2

def simY(x2):
    return 0.5 * sqrt(3.0) * x2

def euclideanNorm(dxNb, dxCr):
    return sqrt(dxNb ** 2 + dxCr ** 2)

def boundBy(x, a, b):
    return (a <= x) and (x <= b)

def labelAxes(xlabel, ylabel, n):
    plt.axis("off")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, simY(1.05)])

    def plot_ticks(start, stop, tick, angle, n):
        plt.text(0.5, -0.075, xlabel, fontsize=18)
        plt.text(simX(-0.11, 0.55), simY(0.55), ylabel, rotation=60, fontsize=18)

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
                    **alignment
                )
        else:
            midx = 0.5 * (start[0] + stop[0])
            midy = 0.5 * (start[1] + stop[1])
            plt.text(midx + dx, midy + dy, "0.5", rotation=angle, **alignment)
        x = np.vstack((x, x + tick[0]))
        y = np.vstack((y, y + tick[1]))
        plt.plot(x, y, "k", lw=1)

    # Spatial considerations
    tick_size = 0.075
    left = np.r_[0, 0]
    right = np.r_[1, 0]
    top = np.r_[simX(0, 1), simY(1)]

    # define vectors for ticks
    bottom_tick = tick_size * np.r_[0, -1] / n
    right_tick = sqrt(3) * tick_size * np.r_[1, 0.5] * (top - left) / n
    left_tick = sqrt(3) * tick_size * np.r_[1, 0.5] * (top - right) / n

    XS = (0, simX(1, 0), simX(0, 1), 0)
    YS = (0, simY(0), simY(1), 0)
    plt.plot(XS, YS, "-k", zorder=2)
    plot_ticks(left, right, bottom_tick, 0, n)
    plot_ticks(right, top, right_tick, -60, n)
    plot_ticks(left, top, left_tick, 60, n)


# === Prepare Axes ===

pltsize = 10
fig1 = plt.figure(figsize=(pltsize, 0.5 * sqrt(3.0) * pltsize))
fig2 = plt.figure(figsize=(pltsize, 0.5 * sqrt(3.0) * pltsize))

plt.figure(1)
plt.title("Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18)
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)

plt.figure(2)
plt.title("Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18)
labelAxes(r"$w_{\mathrm{Nb}}$", r"$w_{\mathrm{Cr}}$", 10)

## === Label Corners of Coexistence Triangle ===

plt.figure(1)
xlGCr = 0.53
xlGNb = 0
xlGNi = 1 - xlGCr - xlGNb
plt.text(simX(xlGNb, xlGCr), simY(xlGCr), "$\gamma$",
    color=colors[0], fontsize=16, zorder=2, **alignment)
xlDCr =-0.006
xlDNb = 0.265
xlDNi = 1 - xlDCr - xlDNb
plt.text(xlDNb, xlDCr, "$\delta$",
         color=colors[1], fontsize=16, zorder=2, **alignment)
xlLCr = 0.35
xlLNb = 0.29
xlLNi = 1 - xlLCr - xlLNb
plt.text(simX(xlLNb, xlLCr), simY(xlLCr), "$\lambda$",
    color=colors[2], fontsize=16, zorder=2, **alignment)

plt.figure(2)
wlGCr, wlGNb, wlGNi = wt_frac(xlGCr, xlGNb, xlGNi)
plt.text(simX(wlGNb, wlGCr), simY(wlGCr), "$\gamma$",
    color=colors[0], fontsize=16, zorder=2, **alignment)
wlDCr, wlDNb, wlDNi = wt_frac(xlDCr, xlDNb, xlDNi)
plt.text(wlDNb, wlDCr, "$\delta$",
         color=colors[1], fontsize=16, zorder=2, **alignment)
wlLCr, wlLNb, wlLNi = wt_frac(xlLCr, xlLNb, xlLNi)
plt.text(simX(wlLNb, wlLCr), simY(wlLCr), "$\lambda$",
    color=colors[2], fontsize=16, zorder=2, **alignment)

plt.figure(1)
xCrMat = (matrixMinCr, matrixMaxCr)
xNbMat = (matrixMinNb, matrixMaxNb)
xMatLbl = xNbMat[0] - 0.009
yMatLbl = 0.5 * (xCrMat[0] + xCrMat[1]) + 0.005

xCrEnr = (enrichMinCr, enrichMaxCr)
xNbEnr = (enrichMinNb, enrichMaxNb)
xEnrLbl = xNbEnr[0] - 0.009
yEnrLbl = 0.5 * (xCrEnr[0] + xCrEnr[1]) + 0.0025

XM = (simX(xNbMat[0], xCrMat[0]),
      simX(xNbMat[0], xCrMat[1]),
      simX(xNbMat[1], xCrMat[1]),
      simX(xNbMat[1], xCrMat[0]),
      simX(xNbMat[0], xCrMat[0]))
YM = (simY(xCrMat[0]),
      simY(xCrMat[1]),
      simY(xCrMat[1]),
      simY(xCrMat[0]),
      simY(xCrMat[0]))

XE = (simX(xNbEnr[0], xCrEnr[0]),
      simX(xNbEnr[0], xCrEnr[1]),
      simX(xNbEnr[1], xCrEnr[1]),
      simX(xNbEnr[1], xCrEnr[0]),
      simX(xNbEnr[0], xCrEnr[0]))
YE = (simY(xCrEnr[0]),
      simY(xCrEnr[1]),
      simY(xCrEnr[1]),
      simY(xCrEnr[0]),
      simY(xCrEnr[0]))

plt.fill(XM, YM, color=salmon, lw=1)
plt.text(
    simX(xMatLbl, yMatLbl),
    simY(yMatLbl),
    "matrix",
    rotation=60,
    fontsize=8,
    **alignment
)
plt.fill(XE, YE, color=rust, lw=1)
plt.text(
    simX(xEnrLbl, yEnrLbl),
    simY(yEnrLbl),
    "enrich",
    rotation=60,
    fontsize=8,
    **alignment
)

plt.figure(2)
wCrMat = (matrixMinCr_w, matrixMaxCr_w)
wNbMat = (matrixMinNb_w, matrixMaxNb_w)
xMatLbl = wNbMat[0] - 0.009
yMatLbl = 0.5 * (wCrMat[0] + wCrMat[1]) + 0.005

wCrEnr = (enrichMinCr_w, enrichMaxCr_w)
wNbEnr = (enrichMinNb_w, enrichMaxNb_w)
xEnrLbl = wNbEnr[0] - 0.009
yEnrLbl = 0.5 * (wCrEnr[0] + wCrEnr[1]) + 0.0025

XM = (simX(wNbMat[0], wCrMat[0]),
      simX(wNbMat[0], wCrMat[1]),
      simX(wNbMat[1], wCrMat[1]),
      simX(wNbMat[1], wCrMat[0]),
      simX(wNbMat[0], wCrMat[0]))
YM = (simY(wCrMat[0]),
      simY(wCrMat[1]),
      simY(wCrMat[1]),
      simY(wCrMat[0]),
      simY(wCrMat[0]))

XE = (simX(wNbEnr[0], wCrEnr[0]),
      simX(wNbEnr[0], wCrEnr[1]),
      simX(wNbEnr[1], wCrEnr[1]),
      simX(wNbEnr[1], wCrEnr[0]),
      simX(wNbEnr[0], wCrEnr[0]))
YE = (simY(wCrEnr[0]),
      simY(wCrEnr[1]),
      simY(wCrEnr[1]),
      simY(wCrEnr[0]),
      simY(wCrEnr[0]))

plt.fill(XM, YM, color=salmon, lw=1)
plt.text(
    simX(xMatLbl, yMatLbl),
    simY(yMatLbl),
    "matrix",
    rotation=60,
    fontsize=8,
    **alignment
)
plt.fill(XE, YE, color=rust, lw=1)
plt.text(
    simX(xEnrLbl, yEnrLbl),
    simY(yEnrLbl),
    "enrich",
    rotation=60,
    fontsize=8,
    **alignment
)

# === Define Systems of Equations ===
# Ref: TKR5p234

def ABSolver(x1, x2):
    def system(X):
        x1A, x2A, x1B, x2B = X
        fA = g_gam(x2A, x1A)
        fB = g_del(x2B, x1B)
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        dx1 = x1A - x1B
        dx2 = x2A - x2B
        return [
            dfAdx1 - dfBdx1,
            dfAdx2 - dfBdx2,
            fA - dfAdx1 * dx1 - dfAdx2 * dx2 - fB,
            (x1 - x1B) * dx2 - dx1 * (x2 - x2B),
        ]

    def jacobian(X):
        # Ref: TKR5p309
        x1A, x2A, x1B, x2B = X
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        d2fAdx11 = d2g_gam_dxNbNb()
        d2fAdx12 = d2g_gam_dxNbCr()
        d2fAdx22 = d2g_gam_dxCrCr()
        d2fBdx11 = d2g_del_dxNbNb()
        d2fBdx12 = d2g_del_dxNbCr()
        d2fBdx22 = d2g_del_dxCrCr()
        dx1 = x1A - x1B
        dx2 = x2A - x2B
        return [
            [d2fAdx11, d2fAdx12, -d2fBdx11, -d2fBdx12],
            [d2fAdx12, d2fAdx22, -d2fBdx12, -d2fBdx22],
            [
                -d2fAdx11 * dx1 - d2fAdx12 * dx2,
                -d2fAdx12 * dx1 - d2fAdx22 * dx2,
                dfAdx1 - dfBdx1,
                dfAdx2 - dfBdx2,
            ],
            [x2B - x2, x1 - x1B, x2 - x2A, x1A - x1]
        ]

    # returns the tuple [x1A, x2A, x1B, x2B]
    return fsolve(func=system, x0=[x1, x2, x1, x2], fprime=jacobian)


def ACSolver(x1, x2):
    def system(X):
        x1A, x2A, x1C, x2C = X
        fA = g_gam(x2A, x1A)
        fC = g_lav(x2C, x1C)
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        dx1 = x1A - x1C
        dx2 = x2A - x2C
        return [
            dfAdx1 - dfCdx1,
            dfAdx2 - dfCdx2,
            fA - dfAdx1 * dx1 - dfAdx2 * dx2 - fC,
            (x1 - x1C) * dx2 - dx1 * (x2 - x2C),
        ]

    def jacobian(X):
        # Ref: TKR5p309
        x1A, x2A, x1C, x2C = X
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        d2fAdx11 = d2g_gam_dxNbNb()
        d2fAdx12 = d2g_gam_dxNbCr()
        d2fAdx22 = d2g_gam_dxCrCr()
        d2fCdx11 = d2g_lav_dxNbNb()
        d2fCdx12 = d2g_lav_dxNbCr()
        d2fCdx22 = d2g_lav_dxCrCr()
        dx1 = x1A - x1C
        dx2 = x2A - x2C
        return [
            [d2fAdx11, d2fAdx12, -d2fCdx11, -d2fCdx12],
            [d2fAdx12, d2fAdx22, -d2fCdx12, -d2fCdx22],
            [
                -d2fAdx11 * dx1 - d2fAdx12 * dx2,
                -d2fAdx12 * dx1 - d2fAdx22 * dx2,
                dfAdx1 - dfCdx1,
                dfAdx2 - dfCdx2,
            ],
            [x2C - x2, x1 - x1C, x2 - x2A, x1A - x1]
        ]

    # returns the tuple [x1A, x2A, x1C, x2C]
    return fsolve(func=system, x0=[x1, x2, x1, x2], fprime=jacobian)


def BCSolver(x1, x2):
    def system(X):
        x1B, x2B, x1C, x2C = X
        fB = g_del(x2B, x1B)
        fC = g_lav(x2C, x1C)
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        dx1 = x1B - x1C
        dx2 = x2B - x2C
        return [
            dfBdx1 - dfCdx1,
            dfBdx2 - dfCdx2,
            fB - dfBdx1 * dx1 - dfBdx2 * dx2 - fC,
            (x1 - x1C) * dx2 - dx1 * (x2 - x2C),
        ]

    def jacobian(X):
        # Ref: TKR5p309
        x1B, x2B, x1C, x2C = X
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        d2fBdx11 = d2g_del_dxNbNb()
        d2fBdx12 = d2g_del_dxNbCr()
        d2fBdx22 = d2g_del_dxCrCr()
        d2fCdx11 = d2g_lav_dxNbNb()
        d2fCdx12 = d2g_lav_dxNbCr()
        d2fCdx22 = d2g_lav_dxCrCr()
        dx1 = x1B - x1C
        dx2 = x2B - x2C
        return [
            [d2fBdx11, d2fBdx12, -d2fCdx11, -d2fCdx12],
            [d2fBdx12, d2fBdx22, -d2fCdx12, -d2fCdx22],
            [
                -d2fBdx11 * dx1 - d2fBdx12 * dx2,
                -d2fBdx12 * dx1 - d2fBdx22 * dx2,
                dfBdx1 - dfCdx1,
                dfBdx2 - dfCdx2,
            ],
            [x2C - x2, x1 - x1C, x2 - x2B, x1B - x1]
        ]

    # returns the tuple [x1B, x2B, x1C, x2C]
    return fsolve(func=system, x0=[x1, x2, x1, x2], fprime=jacobian)


def ABCSolver(x1, x2):
    def system(X):
        x1A, x2A, x1B, x2B, x1C, x2C = X
        fA = g_gam(x2A, x1A)
        fB = g_del(x2B, x1B)
        fC = g_lav(x2C, x1C)
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        dx1B = x1A - x1B
        dx1C = x1A - x1C
        dx2B = x2A - x2B
        dx2C = x2A - x2C
        return [
            dfAdx1 - dfBdx1,
            dfAdx1 - dfCdx1,
            dfAdx2 - dfBdx2,
            dfAdx2 - dfCdx2,
            fA - dfAdx1 * dx1B - dfAdx2 * dx2B - fB,
            fA - dfAdx1 * dx1C - dfAdx2 * dx2C - fC,
        ]

    def jacobian(X):
        x1A, x2A, x1B, x2B, x1C, x2C = X
        dfAdx1 = dg_gam_dxNb(x2A, x1A)
        dfAdx2 = dg_gam_dxCr(x2A, x1A)
        dfBdx1 = dg_del_dxNb(x2B, x1B)
        dfBdx2 = dg_del_dxCr(x2B, x1B)
        dfCdx1 = dg_lav_dxNb(x2C, x1C)
        dfCdx2 = dg_lav_dxCr(x2C, x1C)
        d2fAdx11 = d2g_gam_dxNbNb()
        d2fAdx12 = d2g_gam_dxNbCr()
        d2fAdx22 = d2g_gam_dxCrCr()
        d2fBdx11 = d2g_del_dxNbNb()
        d2fBdx12 = d2g_del_dxNbCr()
        d2fBdx22 = d2g_del_dxCrCr()
        d2fCdx11 = d2g_lav_dxNbNb()
        d2fCdx12 = d2g_lav_dxNbCr()
        d2fCdx22 = d2g_lav_dxCrCr()
        dx1B = x1A - x1B
        dx1C = x1A - x1C
        dx2B = x2A - x2B
        dx2C = x2A - x2C
        return [
            [d2fAdx11, d2fAdx12, -d2fBdx11, -d2fBdx12, 0, 0],
            [d2fAdx11, d2fAdx12, -d2fCdx11, -d2fCdx12, 0, 0],
            [d2fAdx12, d2fAdx22, 0, 0, -d2fBdx12, -d2fBdx22],
            [d2fAdx12, d2fAdx22, 0, 0, -d2fCdx12, -d2fCdx22],
            [
                -d2fAdx11 * dx1B - d2fAdx12 * dx2B,
                -d2fAdx12 * dx1B - d2fAdx22 * dx2B,
                dfAdx1 - dfBdx1,
                dfAdx2 - dfBdx2,
                0,
                0
            ],
            [
                -d2fAdx11 * dx1C - d2fAdx12 * dx2C,
                -d2fAdx12 * dx1C - d2fAdx22 * dx2C,
                0,
                0,
                dfAdx1 - dfCdx1,
                dfAdx2 - dfCdx2
            ]
        ]

    # returns the tuple [x1A, x2A, x1B, x2B]
    return fsolve(func=system, x0=[x1, x2, x1, x2, x1, x2], fprime=jacobian)


# === Plot 3-phase coexistence ===

coexist = []
w_coexist = []

for x1test in np.linspace(0.5 / density, 1 - 0.5 / density, density):
    for x2test in np.linspace(
        0.5 / density, 1 - x1test - 0.5 / density, max(1, ceil((1 - x1test) * density))
    ):
        if len(coexist) < 1:
            x1ABC, x2ABC, x1BAC, x2BAC, x1CAB, x2CAB = ABCSolver(x1test, x2test)
            x3ABC = 1 - x1ABC - x2ABC
            x3BAC = 1 - x1BAC - x2BAC
            x3CAB = 1 - x1CAB - x2CAB

            ABCisPhysical = (
                boundBy(x1ABC, 0, 1)
                and boundBy(x2ABC, 0, 1)
                and boundBy(x3ABC, 0, 1)
                and boundBy(x1BAC, 0, 1)
                and boundBy(x2BAC, 0, 1)
                and boundBy(x3BAC, 0, 1)
                and boundBy(x1CAB, 0, 1)
                and boundBy(x2CAB, 0, 1)
                and boundBy(x3CAB, 0, 1)
                and boundBy(
                    x1test, min((x1ABC, x1BAC, x1CAB)), max((x1ABC, x1BAC, x1CAB))
                )
                and boundBy(
                    x2test, min((x2ABC, x2BAC, x2CAB)), max((x2ABC, x2BAC, x2CAB))
                )
                and boundBy(
                    x1test, min((xe1A(), xe1B(), xe1C())), max((xe1A(), xe1B(), xe1C()))
                )
                and boundBy(
                    x2test, min((xe2A(), xe2B(), xe2C())), max((xe2A(), xe2B(), xe2C()))
                )
            )

            if ABCisPhysical:
                #       gamma corner        delta corner        Laves corner        gamma corner
                triX = (
                    simX(x1ABC, x2ABC),
                    simX(x1BAC, x2BAC),
                    simX(x1CAB, x2CAB),
                    simX(x1ABC, x2ABC),
                )
                triY = (simY(x2ABC), simY(x2BAC), simY(x2CAB), simY(x2ABC))
                coexist.append((triX, triY))
                w2ABC, w1ABC, w3ABC = wt_frac(x2ABC, x1ABC, x3ABC)
                w2BAC, w1BAC, w3BAC = wt_frac(x2BAC, x1BAC, x3BAC)
                w2CAB, w1CAB, w3CAB = wt_frac(x2CAB, x1CAB, x3CAB)
                triX = (
                    simX(w1ABC, w2ABC),
                    simX(w1BAC, w2BAC),
                    simX(w1CAB, w2CAB),
                    simX(w1ABC, w2ABC),
                )
                triY = (simY(w2ABC), simY(w2BAC), simY(w2CAB), simY(w2ABC))
                w_coexist.append((triX, triY))

plt.figure(1)
for x, y in coexist:
    plt.plot(x, y, color="black", zorder=2)
plt.figure(2)
for x, y in w_coexist:
    plt.plot(x, y, color="black", zorder=2)

sAB = []
sAC = []
sBC = []
sBA = []
sCA = []
sCB = []
mAB = []
mAC = []
mBC = []
mBA = []
mCA = []
mCB = []

for x1test in tqdm(np.linspace(0.5 / density, 1 - 0.5 / density, density)):
    for x2test in np.linspace(
        0.5 / density, 1 - x1test - 0.5 / density, max(1, ceil((1 - x1test) * density))
    ):
        x1AB, x2AB, x1BA, x2BA = ABSolver(x1test, x2test)
        x1AC, x2AC, x1CA, x2CA = ACSolver(x1test, x2test)
        x1BC, x2BC, x1CB, x2CB = BCSolver(x1test, x2test)

        x3AB = 1 - x1AB - x2AB
        x3BA = 1 - x1BA - x2BA
        x3AC = 1 - x1AC - x2AC
        x3CA = 1 - x1CA - x2CA
        x3BC = 1 - x1BC - x2BC
        x3CB = 1 - x1CB - x2CB

        ABisPhysical = (
            boundBy(x1AB, 0, 0.51)
            and boundBy(x2AB, 0, 1)
            and boundBy(x3AB, 0, 1)
            and boundBy(x1BA, 0, 0.51)
            and boundBy(x2BA,-0.02, 1)
            and boundBy(x3BA, 0, 1)
            and boundBy(x1test, min(x1AB, x1BA), max(x1AB, x1BA))
            and boundBy(x2test, min(x2AB, x2BA), max(x2AB, x2BA))
        )
        ACisPhysical = (
            boundBy(x1AC, 0, 0.51)
            and boundBy(x2AC, 0, 1)
            and boundBy(x3AC, 0, 1)
            and boundBy(x1CA, 0, 0.51)
            and boundBy(x2CA, 0, 1)
            and boundBy(x3CA, 0, 1)
            and boundBy(x1test, min(x1AC, x1CA), max(x1AC, x1CA))
            and boundBy(x2test, min(x2AC, x2CA), max(x2AC, x2CA))
        )
        BCisPhysical = (
            boundBy(x1BC, 0, 0.51)
            and boundBy(x2BC,-0.1, 1)
            and boundBy(x3BC, 0, 1)
            and boundBy(x1CB, 0, 0.51)
            and boundBy(x2CB, 0, 1)
            and boundBy(x3CB, 0, 1)
            and boundBy(x1test, min(x1BC, x1CB), max(x1BC, x1CB))
            and boundBy(x2test, min(x2BC, x2CB), max(x2BC, x2CB))
        )

        # Compute system energies

        fA = g_gam(x2test, x1test)
        fB = g_del(x2test, x1test)
        fC = g_lav(x2test, x1test)

        fAB = 1e6 * d2g_gam_dxNbNb()
        fAC = 1e6 * d2g_lav_dxNbNb()
        fBC = 1e6 * d2g_del_dxNbNb()

        if ABisPhysical:
            lAB = euclideanNorm(x1BA - x1AB, x2BA - x2AB)
            wA = euclideanNorm(x1BA - x1test, x2BA - x2test) / lAB
            wB = euclideanNorm(x1test - x1AB, x2test - x2AB) / lAB
            fAB = wA * g_gam(x2AB, x1AB) + wB * g_del(x2BA, x1BA)

        if ACisPhysical:
            lAC = euclideanNorm(x1CA - x1AC, x2CA - x2AC)
            wA = euclideanNorm(x1CA - x1test, x2CA - x2test) / lAC
            wC = euclideanNorm(x1test - x1AC, x2test - x2AC) / lAC
            fAC = wA * g_gam(x2AC, x1AC) + wC * g_lav(x2CA, x1CA)

        if BCisPhysical:
            lBC = euclideanNorm(x1CB - x1BC, x2CB - x2BC)
            wB = euclideanNorm(x1CB - x1test, x2CB - x2test) / lBC
            wC = euclideanNorm(x1test - x1BC, x2test - x2BC) / lBC
            fBC = wB * g_del(x2BC, x1BC) + wC * g_lav(x2CB, x1CB)

        energies = np.asarray((fAB, fAC, fBC, fA, fB, fC))
        minIdx = np.argmin(energies)

        if minIdx == 0:
            w2AB, w1AB, w3AB = wt_frac(x2AB, x1AB, x3AB)
            w2BA, w1BA, w3BA = wt_frac(x2BA, x1BA, x3BA)
            xa = (simX(x1AB, x2AB), simY(x2AB))
            xb = (simX(x1BA, x2BA), simY(x2BA))
            wa = (simX(w1AB, w2AB), simY(w2AB))
            wb = (simX(w1BA, w2BA), simY(w2BA))
            if (boundBy(xa[1], 0, simY(xe_gam_Cr))
            and boundBy(xb[1],-0.02, simY(xe_del_Cr))):
                sAB.append(xa)
                sBA.append(xb)
                plt.figure(1)
                plt.scatter(xa[0], xa[1], c=colors[0],
                    marker="h", edgecolor=colors[0], s=1.5, zorder=1)
                plt.scatter(xb[0], xb[1], c=colors[1],
                    marker="h", edgecolor=colors[1], s=1.5, zorder=1)
                plt.plot([xa[0], xb[0]], [xa[1], xb[1]],
                         color="gray", linewidth=0.1, zorder=0)
                plt.figure(2)
                plt.scatter(wa[0], wa[1], c=colors[0],
                    marker="h", edgecolor=colors[0], s=1.5, zorder=1)
                plt.scatter(wb[0], wb[1], c=colors[1],
                    marker="h", edgecolor=colors[1], s=1.5, zorder=1)
                plt.plot([wa[0], wb[0]], [wa[1], wb[1]],
                         color="gray", linewidth=0.1, zorder=0)
            else:
                mAB.append(xa)
                mBA.append(xb)
                """
                plt.figure(1)
                plt.scatter(xa[0], xa[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(xb[0], xb[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.figure(2)
                plt.scatter(wa[0], wa[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(wb[0], wb[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                """
        elif minIdx == 1:
            w2AC, w1AC, w3AC = wt_frac(x2AC, x1AC, x3AC)
            w2CA, w1CA, w3CA = wt_frac(x2CA, x1CA, x3CA)
            xa = (simX(x1AC, x2AC), simY(x2AC))
            xc = (simX(x1CA, x2CA), simY(x2CA))
            wa = (simX(w1AC, w2AC), simY(w2AC))
            wc = (simX(w1CA, w2CA), simY(w2CA))
            if (boundBy(xa[1], simY(xe_gam_Cr), 1)
            and boundBy(xc[1], simY(xe_lav_Cr), 1)):
                sAC.append(xa)
                sCA.append(xc)
                plt.figure(1)
                plt.scatter(xa[0], xa[1], c=colors[0],
                    marker="h", edgecolor=colors[0], s=1.5, zorder=1)
                plt.scatter(xc[0], xc[1], c=colors[2],
                    marker="h", edgecolor=colors[2], s=1.5, zorder=1)
                plt.plot([xa[0], xc[0]], [xa[1], xc[1]],
                         color="gray", linewidth=0.1, zorder=0)
                plt.figure(2)
                plt.scatter(wa[0], wa[1], c=colors[0],
                    marker="h", edgecolor=colors[0], s=1.5, zorder=1)
                plt.scatter(wc[0], wc[1], c=colors[2],
                    marker="h", edgecolor=colors[2], s=1.5, zorder=1)
                plt.plot([wa[0], wc[0]], [wa[1], wc[1]],
                         color="gray", linewidth=0.1, zorder=0)
            else:
                mAC.append(xa)
                mCA.append(xc)
                """
                plt.figure(1)
                plt.scatter(xa[0], xa[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(xc[0], xc[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.figure(2)
                plt.scatter(wa[0], wa[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(wc[0], wc[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                """
        elif minIdx == 2:
            w2BC, w1BC, w3BC = wt_frac(x2BC, x1BC, x3BC)
            w2CB, w1CB, w3CB = wt_frac(x2CB, x1CB, x3CB)
            xb = (simX(x1BC, x2BC), simY(x2BC))
            xc = (simX(x1CB, x2CB), simY(x2CB))
            wb = (simX(w1BC, w2BC), simY(w2BC))
            wc = (simX(w1CB, w2CB), simY(w2CB))
            if (boundBy(xb[0], 0, simX(xe_del_Nb, xe_del_Cr))
            and boundBy(xb[1],-0.1, simY(xe_del_Cr))
            and boundBy(x1CB, xe_del_Nb, 1)
            and boundBy(xc[1], 0, simY(xe_lav_Cr))):
                sBC.append(xb)
                sCB.append(xc)
                plt.figure(1)
                plt.scatter(xb[0], xb[1], c=colors[1],
                    marker="h", edgecolor=colors[1], s=1.5, zorder=1)
                plt.scatter(xc[0], xc[1], c=colors[2],
                    marker="h", edgecolor=colors[2], s=1.5, zorder=1)
                plt.plot([xb[0], xc[0]], [xb[1], xc[1]],
                         color="gray", linewidth=0.1, zorder=0)
                plt.figure(2)
                plt.scatter(wb[0], wb[1], c=colors[1],
                    marker="h", edgecolor=colors[1], s=1.5, zorder=1)
                plt.scatter(wc[0], wc[1], c=colors[2],
                    marker="h", edgecolor=colors[2], s=1.5, zorder=1)
                plt.plot([wb[0], wc[0]], [wb[1], wc[1]],
                         color="gray", linewidth=0.1, zorder=0)
            else:
                mBC.append(xb)
                mCB.append(xc)
                """
                plt.figure(1)
                plt.scatter(xb[0], xb[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(xc[0], xc[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.figure(2)
                plt.scatter(wb[0], wb[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                plt.scatter(wc[0], wc[1], marker="h", c=colors[3],
                            edgecolor=colors[3], s=1.0, zorder=0)
                """

# === Save Image ===

plt.figure(1)
plt.savefig("ternary-diagram.png", dpi=400, bbox_inches="tight")
plt.figure(2)
plt.savefig("weight-diagram.png", dpi=400, bbox_inches="tight")
plt.figure(1)
plt.close()
plt.figure(2)
plt.close()

np.savez_compressed(
    "tie-lines.npz",
    np.asarray(sAB),
    np.asarray(sAC),
    np.asarray(sBC),
    np.asarray(sBA),
    np.asarray(sCA),
    np.asarray(sCB),
)

np.savez_compressed(
    "metastable-lines.npz",
    np.asarray(mAB),
    np.asarray(mAC),
    np.asarray(mBC),
    np.asarray(mBA),
    np.asarray(mCA),
    np.asarray(mCB),
)
