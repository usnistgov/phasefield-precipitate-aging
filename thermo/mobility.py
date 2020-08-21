# -*- coding: utf-8 -*-

import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from constants import *
from pyCinterface import *

xspan = (0.0, 1.0)
yspan = (0.0, 0.95)
nfun = 4
npts = 400
ncon = npts / 2

x = np.linspace(xspan[0], xspan[1], npts)
y = np.linspace(yspan[0], yspan[1], npts)

# Triangular grid
XG = [[]]
YG = [[]]
for a in np.arange(0, 1, 0.1):
    # x1--x2: lines of constant x2=a
    XG.append([simX(a, 0), simX(a, 1-a)])
    YG.append([simY(0),    simY(1-a)])
    # x2--x3: lines of constant x3=a
    XG.append([simX(0, a), simX(1-a, a)])
    YG.append([simY(a),    simY(a)])
    # x1--x3: lines of constant x1=1-a
    XG.append([simX(0, a), simX(a, 0)])
    YG.append([simY(a),    simY(0)])

p = []
q = []

z = []
z.append([])
z.append([])
z.append([])
z.append([])

eig = []
eig.append([])
eig.append([])
eig.append([])
eig.append([])
eig.append([])
eig.append([])

eign = 0
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xCr = 1.0*j / rt3by2
        xNb = 1.0*i - 0.5 * j / rt3by2
        xNi = 1.0 - xCr - xNb
        if xCr > 0. and xNb > 0. and xNi > 0. and xNi < 1.0:
            p.append(i)
            q.append(j)

            # === Diffusivity ===
            D11 = Vm * D_CrCr(xCr, xNb)
            D12 = Vm * D_CrNb(xCr, xNb)
            D21 = Vm * D_NbCr(xCr, xNb)
            D22 = Vm * D_NbNb(xCr, xNb)

            # == Record Values ===
            z[0].append(D11)
            z[1].append(D12)
            z[2].append(D21)
            z[3].append(D22)


# === Diffusivity Plot ===

Titles = (r"$D_{\mathrm{CrCr}}$", r"$D_{\mathrm{CrNb}}$",
          r"$D_{\mathrm{NbCr}}$", r"$D_{\mathrm{NbNb}}$")

f, axarr = plt.subplots(nrows=2, ncols=2)
f.suptitle("Diffusion Constants",fontsize=14)

n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n], fontsize=10)
    ax.set_xlim(xspan)
    ax.set_ylim(yspan)
    ax.axis("equal")
    ax.axis("off")
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ":w", linewidth=0.5)
    ax.plot(XS, YS, "k", linewidth=0.5)
    ax.plot(X0, Y0, "k", linewidth=0.5)
    lvl = ax.tricontourf(p, q, z[n], cmap = plt.cm.get_cmap("coolwarm"))
    cax = plt.colorbar(lvl, ax = ax)
    with warnings.catch_warnings():
        # If there is no zero-level set, that's great! Ignore the warning.
        warnings.simplefilter("ignore")
        lvl = ax.tricontour(p, q, z[n], colors='k', linewidths=0.5, levels=[0.0])
    n += 1

f.savefig("../diagrams/diffusivity.png", dpi=400, bbox_inches="tight")
plt.close()
