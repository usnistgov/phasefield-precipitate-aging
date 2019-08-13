# -*- coding: utf-8 -*-

# Generate phase diagrams
# Usage: python phasediagram.py

from math import ceil, fabs, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm
import warnings

from pyCinterface import *
from constants import *

density = 256

colors = ["red", "green", "blue", "gray"]
salmon = "#fa8072"
rust = "#b7410e"

alignment = {"horizontalalignment": "center", "verticalalignment": "center"}
warnings.filterwarnings("ignore", "The iteration is not making good progress")
warnings.filterwarnings(
    "ignore", "The number of calls to function has reached maxfev = 500."
)

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
plt.figure(figsize=(pltsize, 0.5 * sqrt(3.0) * pltsize))
plt.title("Cr-Nb-Ni at {0} K".format(int(temp)), fontsize=18)
labelAxes(r"$x_{\mathrm{Nb}}$", r"$x_{\mathrm{Cr}}$", 10)

plt.text(
    simX(0.05, 0.10),
    simY(0.10),
    "$\gamma$",
    color=colors[0],
    fontsize=16,
    zorder=2,
    **alignment
)
plt.text(0.265, -0.006, "$\delta$", color=colors[1], fontsize=16, zorder=2, **alignment)
plt.text(
    simX(0.29, 0.35),
    simY(0.35),
    "$\lambda$",
    color=colors[2],
    fontsize=16,
    zorder=2,
    **alignment
)

xCrMat = (matrixMinCr, matrixMaxCr)
xNbMat = (matrixMinNb, matrixMaxNb)
xMatLbl = xNbMat[0] - 0.009
yMatLbl = 0.5 * (xCrMat[0] + xCrMat[1]) + 0.005

xCrEnr = (enrichMinCr, enrichMaxCr)
xNbEnr = (enrichMinNb, enrichMaxNb)
xEnrLbl = xNbEnr[0] - 0.009
yEnrLbl = 0.5 * (xCrEnr[0] + xCrEnr[1]) + 0.0025

XM = (
    simX(xNbMat[0], xCrMat[0]),
    simX(xNbMat[0], xCrMat[1]),
    simX(xNbMat[1], xCrMat[1]),
    simX(xNbMat[1], xCrMat[0]),
    simX(xNbMat[0], xCrMat[0]),
)
YM = (
    simY(xCrMat[0]),
    simY(xCrMat[1]),
    simY(xCrMat[1]),
    simY(xCrMat[0]),
    simY(xCrMat[0]),
)
XE = (
    simX(xNbEnr[0], xCrEnr[0]),
    simX(xNbEnr[0], xCrEnr[1]),
    simX(xNbEnr[1], xCrEnr[1]),
    simX(xNbEnr[1], xCrEnr[0]),
    simX(xNbEnr[0], xCrEnr[0]),
)
YE = (
    simY(xCrEnr[0]),
    simY(xCrEnr[1]),
    simY(xCrEnr[1]),
    simY(xCrEnr[0]),
    simY(xCrEnr[0]),
)

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
            fA + dfAdx1 * dx1 + dfAdx2 * dx2 - fB,
            (x1 - x1B) * dx2 - dx1 * (x2 - x2B),
        ]

    def jacobian(X):
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
                d2fAdx11 * dx1 + 2 * dfAdx1 + d2fAdx12 * dx2,
                d2fAdx12 * dx1 + 2 * dfAdx2 + d2fAdx22 * dx2,
                -dfBdx1 - dfAdx1,
                -dfBdx2 - dfAdx2,
            ],
            [-x2 + x2B, x1 - x1B, -x2A + x2, -x1 + x1A],
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
            fA + dfAdx1 * dx1 + dfAdx2 * dx2 - fC,
            (x1 - x1C) * dx2 - dx1 * (x2 - x2C),
        ]

    def jacobian(X):
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
                d2fAdx11 * dx1 + 2 * dfAdx1 + d2fAdx12 * dx2,
                d2fAdx12 * dx1 + 2 * dfAdx2 + d2fAdx22 * dx2,
                -dfCdx1 - dfAdx1,
                -dfCdx2 - dfAdx2,
            ],
            [-x2 + x2C, x1 - x1C, -x2A + x2, -x1 + x1A],
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
            fB + dfBdx1 * dx1 + dfBdx2 * dx2 - fC,
            (x1 - x1C) * dx2 - dx1 * (x2 - x2C),
        ]

    def jacobian(X):
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
                d2fBdx11 * dx1 + 2 * dfBdx1 + d2fBdx12 * dx2,
                d2fBdx12 * dx1 + 2 * dfBdx2 + d2fBdx22 * dx2,
                -dfCdx1 - dfBdx1,
                -dfCdx2 - dfBdx2,
            ],
            [-x2 + x2C, x1 - x1C, -x2B + x2, -x1 + x1B],
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
            fA + dfAdx1 * dx1B + dfAdx2 * dx2B - fB,
            fA + dfAdx1 * dx1C + dfAdx2 * dx2C - fC,
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
                d2fAdx11 * dx1B + 2 * dfAdx1 + d2fAdx12 * dx2B,
                d2fAdx12 * dx1B + 2 * dfAdx2 + d2fAdx22 * dx2B,
                -dfBdx1 - dfAdx1,
                -dfBdx2 - dfAdx2,
                0,
                0,
            ],
            [
                d2fAdx11 * dx1C + 2 * dfAdx1 + d2fAdx12 * dx2C,
                d2fAdx12 * dx1C + 2 * dfAdx2 + d2fAdx22 * dx2C,
                0,
                0,
                -dfCdx1 - dfAdx1,
                -dfCdx2 - dfAdx2,
            ],
        ]

    # returns the tuple [x1A, x2A, x1B, x2B]
    return fsolve(func=system, x0=[x1, x2, x1, x2, x1, x2], fprime=jacobian)


# === Plot 3-phase coexistence ===

coexist = []

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

for x, y in coexist:
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
            boundBy(x1AB, 0, 1)
            and boundBy(x2AB, 0, 1)
            and boundBy(x3AB, 0, 1)
            and boundBy(x1BA, 0, 1)
            and boundBy(x2BA, 0, 1)
            and boundBy(x3BA, 0, 1)
            and boundBy(x1test, min(x1AB, x1BA), max(x1AB, x1BA))
            and boundBy(x2test, min(x2AB, x2BA), max(x2AB, x2BA))
        )
        ACisPhysical = (
            boundBy(x1AC, 0, 1)
            and boundBy(x2AC, 0, 1)
            and boundBy(x3AC, 0, 1)
            and boundBy(x1CA, 0, 1)
            and boundBy(x2CA, 0, 1)
            and boundBy(x3CA, 0, 1)
            and boundBy(x1test, min(x1AC, x1CA), max(x1AC, x1CA))
            and boundBy(x2test, min(x2AC, x2CA), max(x2AC, x2CA))
        )
        BCisPhysical = (
            boundBy(x1BC, 0, 1)
            and boundBy(x2BC, 0, 1)
            and boundBy(x3BC, 0, 1)
            and boundBy(x1CB, 0, 1)
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
            a = (simX(x1AB, x2AB), simY(x2AB))
            b = (simX(x1BA, x2BA), simY(x2BA))
            if boundBy(a[1], 0, coexist[0][1][0]) and boundBy(
                b[1], 0, coexist[0][1][1]
            ):
                plt.scatter(
                    a[0],
                    a[1],
                    c=colors[0],
                    marker="h",
                    edgecolor=colors[0],
                    s=1.5,
                    zorder=1,
                )
                plt.scatter(
                    b[0],
                    b[1],
                    c=colors[1],
                    marker="h",
                    edgecolor=colors[1],
                    s=1.5,
                    zorder=1,
                )
                plt.plot(
                    [a[0], b[0]], [a[1], b[1]], color="gray", linewidth=0.1, zorder=0
                )
                sAB.append(a)
                sBA.append(b)
            else:
                plt.scatter(
                    a[0],
                    a[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                plt.scatter(
                    b[0],
                    b[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                mAB.append(a)
                mBA.append(b)

        elif minIdx == 1:
            a = (simX(x1AC, x2AC), simY(x2AC))
            c = (simX(x1CA, x2CA), simY(x2CA))
            if boundBy(a[1], coexist[0][1][0], 1) and boundBy(
                c[1], coexist[0][1][2], 1
            ):
                plt.scatter(
                    a[0],
                    a[1],
                    c=colors[0],
                    marker="h",
                    edgecolor=colors[0],
                    s=1.5,
                    zorder=1,
                )
                plt.scatter(
                    c[0],
                    c[1],
                    c=colors[2],
                    marker="h",
                    edgecolor=colors[2],
                    s=1.5,
                    zorder=1,
                )
                plt.plot(
                    [a[0], c[0]], [a[1], c[1]], color="gray", linewidth=0.1, zorder=0
                )
                sAC.append(a)
                sCA.append(c)
            else:
                plt.scatter(
                    a[0],
                    a[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                plt.scatter(
                    c[0],
                    c[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                mAC.append(a)
                mCA.append(c)

        elif minIdx == 2:
            b = (simX(x1BC, x2BC), simY(x2BC))
            c = (simX(x1CB, x2CB), simY(x2CB))
            if boundBy(b[0], coexist[0][0][1], 0.5) and boundBy(
                c[0], coexist[0][0][2], 0.5
            ):
                plt.scatter(
                    b[0],
                    b[1],
                    c=colors[1],
                    marker="h",
                    edgecolor=colors[1],
                    s=1.5,
                    zorder=1,
                )
                plt.scatter(
                    c[0],
                    c[1],
                    c=colors[2],
                    marker="h",
                    edgecolor=colors[2],
                    s=1.5,
                    zorder=1,
                )
                plt.plot(
                    [b[0], c[0]], [b[1], c[1]], color="gray", linewidth=0.1, zorder=0
                )
                sBC.append(b)
                sCB.append(c)
            else:
                plt.scatter(
                    b[0],
                    b[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                plt.scatter(
                    c[0],
                    c[1],
                    marker="h",
                    c=colors[3],
                    edgecolor=colors[3],
                    s=1.5,
                    zorder=0,
                )
                mBC.append(b)
                mCB.append(c)

# === Overlay composition pathway ===


def plotEnrichment(xCrM, xNbM, xCrE, xNbE):
    xlo, xhi = (-0.5e-6, 0.5e-6)
    wCr, wNb = (150e-9, 50e-9)
    pos = np.linspace(xlo, xhi, 256, dtype=float)
    bCr = np.empty_like(pos)
    bNb = np.empty_like(pos)
    for i in range(len(pos)):
        bCr[i] = bellCurve(xlo, xhi, wCr, pos[i], xCrM, xCrE)
        bNb[i] = bellCurve(xlo, xhi, wNb, pos[i], xNbM, xNbE)
    x = simX(bNb, bCr)
    y = simY(bCr)
    plt.plot(x, y, "-r", linewidth=1, zorder=0)


plotEnrichment(matrixMinCr, matrixMinNb, enrichMinCr, enrichMaxNb)

plotEnrichment(matrixMaxCr, matrixMinNb, enrichMaxCr, enrichMaxNb)

# === Save Image ===

plt.savefig("ternary-diagram.png", dpi=400, bbox_inches="tight")

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

plt.close()
