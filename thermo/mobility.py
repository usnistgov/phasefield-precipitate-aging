# -*- coding: utf-8 -*-

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

            # === Mobility ===
            mCC = M_CrCr(xCr, xNb)
            mCN = M_CrNb(xCr, xNb)
            mNN = M_NbNb(xCr, xNb)

            # === Diffusivity ===
            D11 = Vm * (mCC * d2g_gam_dxCrCr() + mCN * d2g_gam_dxCrNb())
            D12 = Vm * (mCC * d2g_gam_dxNbCr() + mCN * d2g_gam_dxNbNb())
            D21 = Vm * (mCN * d2g_gam_dxCrCr() + mNN * d2g_gam_dxCrNb())
            D22 = Vm * (mCN * d2g_gam_dxNbCr() + mNN * d2g_gam_dxNbNb())

            # == Record Values ===
            z[0].append(D11)
            z[1].append(D12)
            z[2].append(D21)
            z[3].append(D22)

            # === Eigenvalues ===
            D = np.array([[D11, D12], [D21, D22]])
            vals, vecs = np.linalg.eig(D)
            n = 0
            for v in vals:
                eig[n].append(v)
                n += 1
            for vec in vecs:
                for v in vec:
                    eig[n].append(v)
                    n += 1
            eign = max(eign, n)

print("Eigen list contains {0} elements".format(eign))

# === Diffusivity Plot ===

Titles = (r"$D_{\mathrm{CrCr}}$", r"$D_{\mathrm{CrNb}}$",
          r"$D_{\mathrm{NbCr}}$", r"$D_{\mathrm{NbNb}}$")

f, axarr = plt.subplots(nrows=2, ncols=2)
f.suptitle("Diffusion Constants",fontsize=14)

n=0
for ax in axarr.reshape(-1):
    # levels = np.logspace(np.log2(xmin), np.log2(xmax), num=ncon, base=2.0)
    ax.set_title(Titles[n], fontsize=10)
    ax.set_xlim(xspan)
    ax.set_ylim(yspan)
    ax.axis("equal")
    ax.axis("off")
    for a in range(len(XG)):
        ax.plot(XG[a], YG[a], ":w", linewidth=0.5)
    ax.plot(XS, YS, "k", linewidth=0.5)
    ax.plot(X0, Y0, "k", linewidth=0.5)
    # ax.tricontourf(p, q, z[n]-datmin[n]+xmin, levels, cmap=plt.cm.get_cmap("coolwarm"), norm=LogNorm())
    lvl = ax.tricontourf(p, q, z[n], cmap = plt.cm.get_cmap("coolwarm"))
    cax = plt.colorbar(lvl, ax = ax)
    lvl = ax.tricontour(p, q, z[n], colors='k', linewidths=0.5, levels=[0.0])
    n += 1

f.savefig("../diagrams/diffusivity.png", dpi=400, bbox_inches="tight")
plt.close()

# === Eigenvalue Plot ===

Titles = (r"$||\lambda_{1}||$", r"$||\lambda_{2}||$",
          r"$\lambda_{11}$", r"$\lambda_{12}$",
          r"$\lambda_{21}$", r"$\lambda_{22}$")

f, axarr = plt.subplots(nrows=1, ncols=2)
f.suptitle("Eigenvalues",fontsize=14)

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
    lvl = ax.tricontourf(p, q, eig[n], cmap = plt.cm.get_cmap("coolwarm"))
    cax = plt.colorbar(lvl, ax = ax)
    lvl = ax.tricontour(p, q, eig[n], colors='k', linewidths=0.5, levels=[0.0])
    n += 1

f.savefig("../diagrams/diffusivity_eigenvalues.png", dpi=400, bbox_inches="tight")
plt.close()

# === Eigenvector Plot ===

f, axarr = plt.subplots(nrows=2, ncols=2)
f.suptitle("Eigenvectors",fontsize=14)

n=2
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
    lvl = ax.tricontourf(p, q, eig[n], cmap = plt.cm.get_cmap("coolwarm"))
    cax = plt.colorbar(lvl, ax = ax)
    lvl = ax.tricontour(p, q, eig[n], colors='k', linewidths=0.5, levels=[0.0])
    n += 1

f.savefig("../diagrams/diffusivity_eigenvectors.png", dpi=400, bbox_inches="tight")
plt.close()
