# -*- coding: utf-8 -*-

"""
Implementation of the ternary solid-state phase-field model in FiPy
"""

import fipy as fp
import fipy.tools.numerix as nx
import time

# === Define Model Parameters ===

from energies import *

# System composition
xCr0 = 0.3125
xNb0 = 0.1575
f0 = 0.125 # initial secondary phase fraction
xCrGam0 = (1 - f0) * xCr0 - f0 / (1 - f0) * xe_del_Cr
xNbGam0 = (1 - f0) * xNb0 - f0 / (1 - f0) * xe_del_Nb

DCrCr =  3.96e-18
DCrNb = -3.40e-17
DNbCr =  3.06e-17
DNbNb =  2.38e-16

sigma = 1.01
w = 2.5e-9
kappa = 3 * sigma * w / 2.2
omega = 18 * sigma**2 / kappa
RT = 9504.68
Vm = 1.0e-5
Ldel = DCrCr * DNbNb / (w**2 * RT * Vm)

# === Setup Computational Domain ===

# Computational domain size
Lx = 1e-6
dx = 5e-9
Nx = Lx / dx

if os.environ.get("FIPY_DISPLAY_MATRIX") == "print":
    Nx = 2

dt = dx**2 / (2.0 * DNbNb)
t = fp.Variable(name="$t$", value=0.)

mesh = fp.Grid1D(dx=dx, nx=Nx)
x = mesh.cellCenters[0]

# === Configure Field Variables ===

phiD = fp.CellVariable(name="$\phi^{\delta}$", mesh=mesh,
                       value=smooth_interface(Lx / 8, 2 * w, x))
phiL = fp.CellVariable(name="$\phi^{\lambda}$", mesh=mesh, value=0.)

pD = fp.CellVariable(name="$p^{\delta}$",  mesh=mesh, value=p(phiD)) # interpolated phi_del
pL = fp.CellVariable(name="$p^{\lambda}$", mesh=mesh, value=p(phiL)) # interpolated phi_lam
pG = fp.CellVariable(name="$p^{\gamma}$",  mesh=mesh, value=1 - pD)  # interpolated phi_gam

xCr = fp.CellVariable(name=r"$x_{\mathrm{Cr}}$", mesh=mesh,
                      value=xCrGam0 * pG + xe_del_Cr * pD)
xNb = fp.CellVariable(name=r"$x_{\mathrm{Nb}}$", mesh=mesh,
                      value=xNbGam0 * pG + xe_del_Nb * pD)

xCrGam = fp.CellVariable(name=r"$x^{\gamma}_{\mathrm{Cr}}$", mesh=mesh,
                         value=x_gam_Cr(xCr, xNb, pD, pG, pL))
xNbGam = fp.CellVariable(name=r"$x^{\gamma}_{\mathrm{Nb}}$", mesh=mesh,
                         value=x_gam_Nb(xCr, xNb, pD, pG, pL))
xCrDel = fp.CellVariable(name=r"$x^{\delta}_{\mathrm{Cr}}$", mesh=mesh,
                         value=x_del_Cr(xCr, xNb, pD, pG, pL))
xNbDel = fp.CellVariable(name=r"$x^{\delta}_{\mathrm{Nb}}$", mesh=mesh,
                         value=x_del_Nb(xCr, xNb, pD, pG, pL))

# === Implement Equations (numbers refer to the manuscript) ===

Agam = 0.5 * d2g_gam_dxCrCr()
Bgam =       d2g_gam_dxCrNb()
Cgam = 0.5 * d2g_gam_dxNbNb()

Adel = 0.5 * d2g_del_dxCrCr()
Bdel =       d2g_del_dxCrNb()
Cdel = 0.5 * d2g_del_dxNbNb()

eq12a = fp.ImplicitSourceTerm(coeff=pG, var=xCrGam) \
      + fp.ImplicitSourceTerm(coeff=pD, var=xCrDel) \
     == fp.ImplicitSourceTerm(coeff=1., var=xCr)

eq12b = fp.ImplicitSourceTerm(coeff=pG, var=xNbGam) \
      + fp.ImplicitSourceTerm(coeff=pD, var=xNbDel) \
     == fp.ImplicitSourceTerm(coeff=1., var=xNb)

eq12c = fp.ImplicitSourceTerm(coeff=Agam, var=xCrGam) \
      + fp.ImplicitSourceTerm(coeff=Bgam, var=xNbGam) \
      - fp.ImplicitSourceTerm(coeff=Adel, var=xCrDel) \
      - fp.ImplicitSourceTerm(coeff=Bdel, var=xNbDel) \
     == xe_gam_Cr * Agam \
      + xe_gam_Nb * Bgam \
      - xe_del_Cr * Adel \
      - xe_del_Nb * Bdel

eq12d = fp.ImplicitSourceTerm(coeff=Bgam, var=xCrGam) \
      + fp.ImplicitSourceTerm(coeff=Cgam, var=xNbGam) \
      - fp.ImplicitSourceTerm(coeff=Bdel, var=xCrDel) \
      - fp.ImplicitSourceTerm(coeff=Cdel, var=xNbDel) \
     == xe_gam_Cr * Bgam \
      + xe_gam_Nb * Cgam \
      - xe_del_Cr * Bdel \
      - xe_del_Nb * Cdel

pressure = p_prime(phiD) * (g_gamma(xCrGam, xNbGam) - g_delta(xCrDel, xNbDel)) \
         - fp.ImplicitSourceTerm(var=xCrGam, coeff=p_prime(phiD) * dg_gam_dxCr(xCrGam, xNbGam)) \
         + fp.ImplicitSourceTerm(var=xCrDel, coeff=p_prime(phiD) * dg_gam_dxCr(xCrGam, xNbGam)) \
         - fp.ImplicitSourceTerm(var=xNbGam, coeff=p_prime(phiD) * dg_gam_dxNb(xCrGam, xNbGam)) \
         + fp.ImplicitSourceTerm(var=xNbDel, coeff=p_prime(phiD) * dg_gam_dxNb(xCrGam, xNbGam))

eq25 = fp.TransientTerm(coeff=1.0/Ldel, var=phiD) \
    == pressure \
     - 2. * omega * phiD * (1. - phiD) * (1. - 2. * phiD) \
     + fp.DiffusionTerm(coeff=kappa, var=phiD)

eq28a = fp.TransientTerm(var=xCr) == fp.DiffusionTerm(coeff=DCrCr * pG, var=xCrGam) \
                                   + fp.DiffusionTerm(coeff=DCrNb * pG, var=xNbGam) \
                                   + fp.DiffusionTerm(coeff=DCrCr * pD, var=xCrDel) \
                                   + fp.DiffusionTerm(coeff=DCrNb * pD, var=xNbDel)

eq28b = fp.TransientTerm(var=xNb) == fp.DiffusionTerm(coeff=DNbCr * pG, var=xCrGam) \
                                   + fp.DiffusionTerm(coeff=DNbNb * pG, var=xNbGam) \
                                   + fp.DiffusionTerm(coeff=DNbCr * pD, var=xCrDel) \
                                   + fp.DiffusionTerm(coeff=DNbNb * pD, var=xNbDel)

coupled = eq12a & eq12b & eq12c & eq12d & eq25 & eq28a & eq28b

# === Turn the Crank ===

viewer = fp.Viewer(vars=(phiD, xCr, xNb, xCrGam, xNbGam), datamin=-0.05, datamax=1.05)
viewer.plot()

fp.input("Initial condition. Press <return> to continue...")
#print("Initial condition. Sleeping 5 sec...")
#time.sleep(5)

for i in range(10):
    print("t =", t.value, "sec")
    #res = coupled.solve(dt=dt)
    #print("    ", res)
    for sweep in range(5):
        res = coupled.sweep(dt=dt)
        print("    ", res)
    t.value = t() + dt
    # i = i + 1
    # if (i % 1000):
    #     viewer.plot()

viewer.plot()
