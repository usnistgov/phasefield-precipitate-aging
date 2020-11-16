# -*- coding: utf-8 -*-

"""
Implementation of the ternary solid-state phase-field model in FiPy
"""

from fipy import Variable, FaceVariable, CellVariable, Grid1D, ImplicitSourceTerm, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix

from energies import *

Lx = 1e-6
dx = 5e-9
nx = Lx / dx

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
    return 0.5 * (1 - numerix.tanh((x - h) / (0.5 * w)))

def p(x):
    return x**3 *(10 - 15 * x + 6 * x**2)
def p_prime(x):
    return 30. * x**2 * (1 - x)**2

mesh = Grid1D(dx=dx, nx=nx)
x = mesh.cellCenters[0]

phiD = CellVariable(name="$\delta$", mesh=mesh, value=pf_tanh(x, Lx / 8))
phiG = CellVariable(name="$\gamma$", mesh=mesh, value=1 - p(phiD))
phiL = CellVariable(name="$\lambda$", mesh=mesh, value=0.)

xCr = CellVariable(name=r"$\x_{\mathrm{Cr}}$", mesh=mesh, value=xe_gam_Cr * phiG + xe_del_Cr * phiD)
xNb = CellVariable(name=r"$\x_{\mathrm{Nb}}$", mesh=mesh, value=xe_gam_Nb * phiG + xe_del_Nb * phiD)
xNi = CellVariable(name=r"$\x_{\mathrm{Ni}}$", mesh=mesh, value=1 - xCr - xNb)

xCrGam = CellVariable(name=r"$\x^{\gamma}_{\mathrm{Cr}}$", mesh=mesh, value=x_gam_Cr(xCr, xNb, phiD, phiG, phiL))
xNbGam = CellVariable(name=r"$\x^{\gamma}_{\mathrm{Nb}}$", mesh=mesh, value=x_gam_Nb(xCr, xNb, phiD, phiG, phiL))
xCrDel = CellVariable(name=r"$\x^{\delta}_{\mathrm{Cr}}$", mesh=mesh, value=x_del_Cr(xCr, xNb, phiD, phiG, phiL))
xNbDel = CellVariable(name=r"$\x^{\delta}_{\mathrm{Nb}}$", mesh=mesh, value=x_del_Nb(xCr, xNb, phiD, phiG, phiL))

# Equation numbers refer to the draft manuscript.

eq12a = ImplicitSourceTerm(coeff=phiG, var=xCrGam) + ImplicitSourceTerm(coeff=phiD, var=xCrDel) == ImplicitSourceTerm(coeff=1, var=xCr)
eq12b = ImplicitSourceTerm(coeff=phiG, var=xNbGam) + ImplicitSourceTerm(coeff=phiD, var=xNbDel) == ImplicitSourceTerm(coeff=1, var=xNb)

eq12c = ImplicitSourceTerm(coeff=0.5*d2g_gam_dxCrCr(), var=xCrGam) + ImplicitSourceTerm(coeff=d2g_gam_dxCrNb(), var=xNbGam) \
      - ImplicitSourceTerm(coeff=0.5*d2g_del_dxCrCr(), var=xCrDel) - ImplicitSourceTerm(coeff=d2g_del_dxCrNb(), var=xNbDel) \
      == 0.5 * xe_gam_Cr * d2g_gam_dxCrCr() + xe_gam_Nb * d2g_gam_dxCrNb() \
       - 0.5 * xe_del_Cr * d2g_del_dxCrCr() - xe_del_Nb * d2g_del_dxCrNb()
eq12d = ImplicitSourceTerm(coeff=d2g_gam_dxCrNb(), var=xCrGam) + ImplicitSourceTerm(coeff=0.5*d2g_gam_dxNbNb(), var=xNbGam) \
      - ImplicitSourceTerm(coeff=d2g_del_dxCrNb(), var=xCrDel) - ImplicitSourceTerm(coeff=0.5*d2g_del_dxNbNb(), var=xNbDel) \
      == xe_gam_Cr * d2g_gam_dxCrNb() + 0.5 * xe_gam_Nb * d2g_gam_dxNbNb() \
       - xe_del_Cr * d2g_del_dxCrNb() - 0.5 * xe_del_Nb * d2g_del_dxNbNb()

pressure = g_gamma(xCrGam, xNbGam) - g_delta(xCrGam, xNbGam) - ImplicitSourceTerm(var=xCrGam, coeff=dg_gam_dxCr(xCr, xNb)) \
                                                             + ImplicitSourceTerm(var=xCrDel, coeff=dg_del_dxCr(xCr, xNb)) \
                                                             - ImplicitSourceTerm(var=xNbGam, coeff=dg_gam_dxNb(xCr, xNb)) \
                                                             + ImplicitSourceTerm(var=xNbDel, coeff=dg_del_dxNb(xCr, xNb))
eq24 = TransientTerm(coeff=1.0/Ldel, var=phiD) == p_prime(phiD) * pressure \
                                                - 2 * omega * phiD * (1 - phiD) * (1 - 2 * phiD) \
                                                + DiffusionTerm(coeff=kappa, var=phiD)

eq27a = TransientTerm(var=xCr) == DiffusionTerm(coeff=DCrCr, var=xCrGam) + DiffusionTerm(coeff=DCrNb, var=xNbGam) \
                                + DiffusionTerm(coeff=DCrCr, var=xCrDel) + DiffusionTerm(coeff=DCrNb, var=xNbDel)
eq27b = TransientTerm(var=xNb) == DiffusionTerm(coeff=DNbCr, var=xCrGam) + DiffusionTerm(coeff=DNbNb, var=xNbGam) \
                                + DiffusionTerm(coeff=DNbCr, var=xCrDel) + DiffusionTerm(coeff=DNbNb, var=xNbDel)

eq = eq12a & eq12b & eq12c & eq12d & eq24 & eq27a & eq27b

# Turn the crank

viewer = Viewer(vars=(phiD, xCr, xNb))
viewer.plot()

dt = 1.0e-9
t = 0
i = 0

while (t < 1):
    eq.solve()
    eq.sweep()
    t += dt
    i += 1
    if (i % 1000):
        viewer.plot()
