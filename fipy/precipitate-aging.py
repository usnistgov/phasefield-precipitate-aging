# -*- coding: utf-8 -*-

"""
Implementation of the ternary solid-state phase-field model in FiPy
"""

import fipy as fp
import fipy.tools.numerix as nx

from energies import *

Lx = 1e-6
dx = 5e-9
Nx = Lx / dx

xCr0 = 0.3125
xNb0 = 0.1575

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

def pf_tanh(x, h):
    return 0.5 * (1 - nx.tanh((x - h) / (0.5 * w)))

mesh = fp.Grid1D(dx=dx, nx=Nx)
x = mesh.cellCenters[0]

phiD = fp.CellVariable(name="$\delta$", mesh=mesh, value=pf_tanh(x, Lx / 8))
phiG = fp.CellVariable(name="$\gamma$", mesh=mesh, value=1 - p(phiD))
phiL = fp.CellVariable(name="$\lambda$", mesh=mesh, value=0.)

xCr = fp.CellVariable(name=r"$x_{\mathrm{Cr}}$", mesh=mesh, value=xe_gam_Cr * phiG + xe_del_Cr * phiD)
xNb = fp.CellVariable(name=r"$x_{\mathrm{Nb}}$", mesh=mesh, value=xe_gam_Nb * phiG + xe_del_Nb * phiD)
xNi = fp.CellVariable(name=r"$x_{\mathrm{Ni}}$", mesh=mesh, value=1 - xCr - xNb)

xCrGam = fp.CellVariable(name=r"$x^{\gamma}_{\mathrm{Cr}}$", mesh=mesh, value=x_gam_Cr(xCr, xNb, phiD, phiG, phiL))
xNbGam = fp.CellVariable(name=r"$x^{\gamma}_{\mathrm{Nb}}$", mesh=mesh, value=x_gam_Nb(xCr, xNb, phiD, phiG, phiL))
xCrDel = fp.CellVariable(name=r"$x^{\delta}_{\mathrm{Cr}}$", mesh=mesh, value=x_del_Cr(xCr, xNb, phiD, phiG, phiL))
xNbDel = fp.CellVariable(name=r"$x^{\delta}_{\mathrm{Nb}}$", mesh=mesh, value=x_del_Nb(xCr, xNb, phiD, phiG, phiL))

# Equation numbers refer to the draft manuscript.

eq12a = fp.ImplicitSourceTerm(coeff=phiG, var=xCrGam) \
      + fp.ImplicitSourceTerm(coeff=phiD, var=xCrDel) \
      == fp.ImplicitSourceTerm(coeff=1, var=xCr)

eq12b = fp.ImplicitSourceTerm(coeff=phiG, var=xNbGam) \
      + fp.ImplicitSourceTerm(coeff=phiD, var=xNbDel) \
      == fp.ImplicitSourceTerm(coeff=1, var=xNb)

eq12c  = fp.ImplicitSourceTerm(coeff=0.5*d2g_gam_dxCrCr(), var=xCrGam) \
       + fp.ImplicitSourceTerm(coeff=d2g_gam_dxCrNb(), var=xNbGam) \
       - fp.ImplicitSourceTerm(coeff=0.5*d2g_del_dxCrCr(), var=xCrDel) \
       - fp.ImplicitSourceTerm(coeff=d2g_del_dxCrNb(), var=xNbDel) \
      == 0.5 * xe_gam_Cr * d2g_gam_dxCrCr() + xe_gam_Nb * d2g_gam_dxCrNb() \
       - 0.5 * xe_del_Cr * d2g_del_dxCrCr() - xe_del_Nb * d2g_del_dxCrNb()

eq12d  = fp.ImplicitSourceTerm(coeff=d2g_gam_dxCrNb(), var=xCrGam) \
       + fp.ImplicitSourceTerm(coeff=0.5*d2g_gam_dxNbNb(), var=xNbGam) \
       - fp.ImplicitSourceTerm(coeff=d2g_del_dxCrNb(), var=xCrDel) \
       - fp.ImplicitSourceTerm(coeff=0.5*d2g_del_dxNbNb(), var=xNbDel) \
      == xe_gam_Cr * d2g_gam_dxCrNb() + 0.5 * xe_gam_Nb * d2g_gam_dxNbNb() \
       - xe_del_Cr * d2g_del_dxCrNb() - 0.5 * xe_del_Nb * d2g_del_dxNbNb()

pressure = p_prime(phiD) * g_gamma(xCrGam, xNbGam) \
         - p_prime(phiD) * g_delta(xCrGam, xNbGam) \
         - fp.ImplicitSourceTerm(var=xCrGam, coeff=p_prime(phiD) * dg_gam_dxCr(xCr, xNb)) \
         + fp.ImplicitSourceTerm(var=xCrDel, coeff=p_prime(phiD) * dg_gam_dxCr(xCr, xNb)) \
         - fp.ImplicitSourceTerm(var=xNbGam, coeff=p_prime(phiD) * dg_gam_dxNb(xCr, xNb)) \
         + fp.ImplicitSourceTerm(var=xNbDel, coeff=p_prime(phiD) * dg_gam_dxNb(xCr, xNb))

eq24 = fp.TransientTerm(coeff=1.0/Ldel, var=phiD) == pressure \
     - 2 * omega * phiD * (1 - phiD) * (1 - 2 * phiD) \
     + fp.DiffusionTerm(coeff=kappa, var=phiD)

eq27a = fp.TransientTerm(var=xCr) == fp.DiffusionTerm(coeff=DCrCr, var=xCrGam) \
                                   + fp.DiffusionTerm(coeff=DCrNb, var=xNbGam) \
                                   + fp.DiffusionTerm(coeff=DCrCr, var=xCrDel) \
                                   + fp.DiffusionTerm(coeff=DCrNb, var=xNbDel)

eq27b = fp.TransientTerm(var=xNb) == fp.DiffusionTerm(coeff=DNbCr, var=xCrGam) \
                                   + fp.DiffusionTerm(coeff=DNbNb, var=xNbGam) \
                                   + fp.DiffusionTerm(coeff=DNbCr, var=xCrDel) \
                                   + fp.DiffusionTerm(coeff=DNbNb, var=xNbDel)

eq = eq12a & eq12b & eq12c & eq12d & eq24 & eq27a & eq27b

# Turn the crank

viewer = fp.Viewer(vars=(phiD, xCr, xNb), datamin=0, datamax=1)
viewer.plot()

dt = 1.0e-9
t = fp.Variable(name="$t$", value=0.)
i = 0

while (t.value < 1):
    # eq.solve(dt=dt)
    for sweep in range(5):
        eq.sweep(dt=dt)
    t.value = t() + dt
    i += 1
    if (i % 1000):
        viewer.plot()
