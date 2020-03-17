# -*- coding: utf-8 -*-

# Gibbs free energy expressions for IN625 from ternary CALPHAD Database
#
# This script extracts relevant thermodynamic functions necessary for the
# phase-field model of solid-state transformations in additively manufactured
# superalloy 625, represented as a ternary (Cr-Nb-Ni) with γ, δ, μ, and Laves
# phases competing. The thermodynamic database was prepared by U. Kattner after
# Du, Liu, Chang, and Yang (2005):
#
# @Article{Du2005,
#     Title   = {A thermodynamic modeling of the Cr–Nb–Ni system },
#     Author  = {Yong Du and Shuhong Liu and Y.A. Chang and Ying Yang},
#     Journal = {Calphad},
#     Year    = {2005},
#     Volume  = {29},
#     Number  = {2},
#     Pages   = {140 - 148},
#     Doi     = {10.1016/j.calphad.2005.06.001}
# }
#
# This database models the phases of interest as follows:
# - γ as $\mathrm{(Cr, Nb, Ni)}$
# - δ as $\mathrm{(\mathbf{Nb}, Ni)_1(Cr, Nb, \mathbf{Ni})_3}$
# - Laves as $\mathrm{(\mathbf{Cr}, Nb, Ni)_2(Cr, \mathbf{Nb})_1}$
#
# The phase field model requires Gibbs free energies as functions of system
# compositions $x_\mathrm{Cr}$, $x_\mathrm{Nb}$, $x_\mathrm{Ni}$. The CALPHAD
# database represents these energies as functions of sublattice compositions
# $y$ in each phase. To avoid solving for internal phase equilibrium at each
# point in the simulation, approximations have been made to allow the following
# one-to-one mappings between $x$ and $y$:
#
# - γ: no changes necessary
#      * $y_\mathrm{Cr}' = x_\mathrm{Cr}$
#      * $y_\mathrm{Nb}' = x_\mathrm{Nb}$
#      * $y_\mathrm{Ni}' = x_\mathrm{Ni}$
#
# - δ: eliminate Nb from the second (Ni) sublattice,
#      $\mathrm{(\mathbf{Nb}, Ni)_1(Cr, \mathbf{Ni})_3}$
#      * $y_\mathrm{Nb}'  = 4x_\mathrm{Nb}$
#      * $y_\mathrm{Ni}'  = 1 - 4x_\mathrm{Nb}$
#      * $y_\mathrm{Cr}'' = \frac{4}{3}x_\mathrm{Cr}$
#      * $y_\mathrm{Ni}'' = 1 - \frac{4}{3}x_\mathrm{Cr}$
#      * Constraints: $x_\mathrm{Nb}\leq\frac{1}{4}$
#                     $x_\mathrm{Cr}\leq\frac{3}{4}$
#
# - Laves: eliminate Nb from the first (Cr) sublattice,
#      $\mathrm{(\mathbf{Cr}, Ni)_2(Cr, \mathbf{Nb})_1}$
#      * $y_\mathrm{Cr}'  = 1 - \frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Ni}'  = \frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Cr}'' = 1 - 3x_\mathrm{Nb}$
#      * $y_\mathrm{Nb}'' = 3x_\mathrm{Nb}$
#      * Constraints: $0\leq x_\mathrm{Ni}\leq\frac{2}{3}$
#                     $0\leq x_\mathrm{Nb}\leq\frac{1}{3}$

# Numerical libraries
import numpy as np

# Thermodynamics and computer-algebra libraries
from pycalphad import Database, calculate, Model
from sympy import Eq, Matrix, diff, expand, init_printing, factor, fraction, pprint, symbols
from sympy.abc import L, r, x, y, z
from sympy.core.numbers import pi
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import tanh
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve, solve_linear_system
from sympy.utilities.codegen import codegen
init_printing()

# Thermodynamic information
from pycalphad import Database, Model
from constants import *

interpolator = x ** 3 * (6.0 * x ** 2 - 15.0 * x + 10.0)
dinterpdx = 30.0 * x ** 2 * (1.0 - x) ** 2
interfaceProfile = (1 - tanh(z)) / 2

# Read CALPHAD database from disk, specify phases and elements of interest
tdb = Database("Du_Cr-Nb-Ni_simple.tdb")
elements = ["CR", "NB", "NI"]

species = list(set([i for c in tdb.phases["FCC_A1"].constituents for i in c]))
model = Model(tdb, species, "FCC_A1")
g_gamma = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases["D0A_NBNI3"].constituents for i in c]))
model = Model(tdb, species, "D0A_NBNI3")
g_delta = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases["C14_LAVES"].constituents for i in c]))
model = Model(tdb, species, "C14_LAVES")
g_laves = parse_expr(str(model.ast))


# Declare sublattice variables used in Pycalphad expressions
XCR, XNB, XNI = symbols("XCR XNB XNI")

# Define lever rule equations
## Ref: TKR4p161, 172; TKR5p266, 272, 293
xo, yo = symbols("xo yo")
xb, yb = symbols("xb yb")
xc, yc = symbols("xc yc")
xd, yd = symbols("xd yd")

levers = solve_linear_system(
    Matrix(
        ((yo - yb, xb - xo, xb * yo - xo * yb), (yc - yd, xd - xc, xd * yc - xc * yd))
    ),
    x,
    y,
)

def draw_bisector(weightA, weightB):
    bNb = (weightA * xe_del_Nb + weightB * xe_lav_Nb) / (weightA + weightB)
    bCr = (weightA * xe_del_Cr + weightB * xe_lav_Cr) / (weightA + weightB)
    xPrime = [simX(xe_gam_Nb, xe_gam_Cr), simX(bNb, bCr)]
    yPrime = [simY(xe_gam_Cr), simY(bCr)]
    return xPrime, yPrime


# Make sublattice -> system substitutions
g_gamma = inVm * g_gamma.subs(
    {
        symbols("FCC_A10CR"): XCR,
        symbols("FCC_A10NB"): XNB,
        symbols("FCC_A10NI"): XNI,
        symbols("FCC_A11VA"): 1,
        symbols("T"): temp,
    }
)

g_delta = inVm * g_delta.subs(
    {
        symbols("D0A_NBNI30NB"): 4 * XNB,
        symbols("D0A_NBNI30NI"): 1 - 4 * XNB,
        symbols("D0A_NBNI31CR"): fr4by3 * XCR,
        symbols("D0A_NBNI31NI"): 1 - fr4by3 * XCR,
        symbols("T"): temp,
    }
)

g_laves = inVm * g_laves.subs(
    {
        symbols("C14_LAVES0CR"): 1 - fr3by2 * (1 - XCR - XNB),
        symbols("C14_LAVES0NI"): fr3by2 * (1 - XCR - XNB),
        symbols("C14_LAVES1CR"): 1 - 3 * XNB,
        symbols("C14_LAVES1NB"): 3 * XNB,
        symbols("T"): temp,
    }
)

# Extract thermodynamic properties from CALPHAD landscape

G_g = Vm * g_gamma

g_dGgam_dxCr = diff(G_g, XCR)
g_dGgam_dxNb = diff(G_g, XNB)
g_dGgam_dxNi = diff(G_g, XNI)

mu_Cr = G_g      - XNB  * g_dGgam_dxNb + (1 - XCR) * g_dGgam_dxCr
mu_Nb = G_g + (1 - XNB) * g_dGgam_dxNb      - XCR  * g_dGgam_dxCr
mu_Ni = G_g      - XNB  * g_dGgam_dxNb      - XCR  * g_dGgam_dxCr

# Simple Curvatures
duCr_dxCr = diff(G_g, XCR, XCR).subs(XNI, 1 - XCR - XNB)
duCr_dxNb = diff(G_g, XCR, XNB).subs(XNI, 1 - XCR - XNB)
duCr_dxNi = diff(G_g, XCR, XNI).subs(XNI, 1 - XCR - XNB)

duNb_dxCr = diff(G_g, XNB, XCR).subs(XNI, 1 - XCR - XNB)
duNb_dxNb = diff(G_g, XNB, XNB).subs(XNI, 1 - XCR - XNB)
duNb_dxNi = diff(G_g, XNB, XNI).subs(XNI, 1 - XCR - XNB)

duNi_dxCr = diff(G_g, XNI, XCR).subs(XNI, 1 - XCR - XNB)
duNi_dxNb = diff(G_g, XNI, XNB).subs(XNI, 1 - XCR - XNB)
duNi_dxNi = diff(G_g, XNI, XNI).subs(XNI, 1 - XCR - XNB)

"""
# Derivatives of Chemical Potentials
duCr_dxCr = diff(mu_Cr, XCR).subs(XNI, 1 - XCR - XNB)
duCr_dxNb = diff(mu_Cr, XNB).subs(XNI, 1 - XCR - XNB)
duCr_dxNi = diff(mu_Cr, XNI).subs(XNI, 1 - XCR - XNB)

duNb_dxCr = diff(mu_Nb, XCR).subs(XNI, 1 - XCR - XNB)
duNb_dxNb = diff(mu_Nb, XNB).subs(XNI, 1 - XCR - XNB)
duNb_dxNi = diff(mu_Nb, XNI).subs(XNI, 1 - XCR - XNB)

duNi_dxCr = diff(mu_Ni, XCR).subs(XNI, 1 - XCR - XNB)
duNi_dxNb = diff(mu_Ni, XNB).subs(XNI, 1 - XCR - XNB)
duNi_dxNi = diff(mu_Ni, XNI).subs(XNI, 1 - XCR - XNB)
"""

# Redefine without solvent species (Ni)

g_gamma = g_gamma.subs(XNI, 1 - XCR - XNB)
G_g = Vm * g_gamma
g_dGgam_dxCr = g_dGgam_dxCr.subs(XNI, 1 - XCR - XNB)
g_dGgam_dxNb = g_dGgam_dxNb.subs(XNI, 1 - XCR - XNB)
g_dGgam_dxNi = g_dGgam_dxNi.subs(XNI, 1 - XCR - XNB)
mu_Cr = mu_Cr.subs(XNI, 1 - XCR - XNB)
mu_Nb = mu_Nb.subs(XNI, 1 - XCR - XNB)
mu_Ni = mu_Ni.subs(XNI, 1 - XCR - XNB)

XNI = 1 - XCR - XNB


# Generate paraboloid expressions (2nd-order Taylor series approximations)

g_d2Gdel_dxCrCr = diff(g_delta, XCR, XCR)
g_d2Gdel_dxCrNb = diff(g_delta, XCR, XNB)
g_d2Gdel_dxNbCr = g_d2Gdel_dxCrNb
g_d2Gdel_dxNbNb = diff(g_delta, XNB, XNB)

g_d2Glav_dxCrCr = diff(g_laves, XCR, XCR)
g_d2Glav_dxCrNb = diff(g_laves, XCR, XNB)
g_d2Glav_dxNbCr = g_d2Glav_dxCrNb
g_d2Glav_dxNbNb = diff(g_laves, XNB, XNB)

# Curvatures
PC_gam_CrCr = duCr_dxCr.subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})
PC_gam_CrNb = duNb_dxCr.subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})
PC_gam_NbNb = duNb_dxNb.subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})

PC_del_CrCr = g_d2Gdel_dxCrCr.subs({XCR: xe_del_Cr, XNB: xe_del_Nb})
PC_del_CrNb = g_d2Gdel_dxCrNb.subs({XCR: xe_del_Cr, XNB: xe_del_Nb})
PC_del_NbNb = g_d2Gdel_dxNbNb.subs({XCR: xe_del_Cr, XNB: xe_del_Nb})

PC_lav_CrCr = g_d2Glav_dxCrCr.subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})
PC_lav_CrNb = g_d2Glav_dxCrNb.subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})
PC_lav_NbNb = g_d2Glav_dxNbNb.subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})

# Expressions
p_gamma = (
    fr1by2 * PC_gam_CrCr * (XCR - xe_gam_Cr) ** 2
    + PC_gam_CrNb * (XCR - xe_gam_Cr) * (XNB - xe_gam_Nb)
    + fr1by2 * PC_gam_NbNb * (XNB - xe_gam_Nb) ** 2
)

p_delta = (
    fr1by2 * PC_del_CrCr * (XCR - xe_del_Cr) ** 2
    + PC_del_CrNb * (XCR - xe_del_Cr) * (XNB - xe_del_Nb)
    + fr1by2 * PC_del_NbNb * (XNB - xe_del_Nb) ** 2
)

p_laves = (
    fr1by2 * PC_lav_CrCr * (XCR - xe_lav_Cr) ** 2
    + PC_lav_CrNb * (XCR - xe_lav_Cr) * (XNB - xe_lav_Nb)
    + fr1by2 * PC_lav_NbNb * (XNB - xe_lav_Nb) ** 2
)

# Generate first derivatives of paraboloid landscape
p_dGgam_dxCr = diff(p_gamma, XCR)
p_dGgam_dxNb = diff(p_gamma, XNB)

p_dGdel_dxCr = diff(p_delta, XCR)
p_dGdel_dxNb = diff(p_delta, XNB)

p_dGlav_dxCr = diff(p_laves, XCR)
p_dGlav_dxNb = diff(p_laves, XNB)

# Generate second derivatives of paraboloid landscape
p_d2Ggam_dxCrCr = diff(p_gamma, XCR, XCR)
p_d2Ggam_dxCrNb = diff(p_gamma, XCR, XNB)
p_d2Ggam_dxNbCr = diff(p_gamma, XNB, XCR)
p_d2Ggam_dxNbNb = diff(p_gamma, XNB, XNB)

p_d2Gdel_dxCrCr = diff(p_delta, XCR, XCR)
p_d2Gdel_dxCrNb = diff(p_delta, XCR, XNB)
p_d2Gdel_dxNbCr = diff(p_delta, XNB, XCR)
p_d2Gdel_dxNbNb = diff(p_delta, XNB, XNB)

p_d2Glav_dxCrCr = diff(p_laves, XCR, XCR)
p_d2Glav_dxCrNb = diff(p_laves, XCR, XNB)
p_d2Glav_dxNbCr = diff(p_laves, XNB, XCR)
p_d2Glav_dxNbNb = diff(p_laves, XNB, XNB)

# ========= FICTITIOUS COMPOSITIONS ==========
# Derivation: TKR4p181
gamCr, gamNb = symbols("gamCr, gamNb")
delCr, delNb = symbols("delCr, delNb")
lavCr, lavNb = symbols("lavCr, lavNb")
pGam, pDel, pLav = symbols("pGam, pDel, pLav")
INV_DET = symbols("INV_DET")
gcd = 1.0e-60

ficGdCr = p_dGgam_dxCr.subs({XCR: gamCr, XNB: gamNb})
ficGdNb = p_dGgam_dxNb.subs({XCR: gamCr, XNB: gamNb})
ficDdCr = p_dGdel_dxCr.subs({XCR: delCr, XNB: delNb})
ficDdNb = p_dGdel_dxNb.subs({XCR: delCr, XNB: delNb})
ficLdCr = p_dGlav_dxCr.subs({XCR: lavCr, XNB: lavNb})
ficLdNb = p_dGlav_dxNb.subs({XCR: lavCr, XNB: lavNb})

ficEqns = (
    XCR - pGam * gamCr - pDel * delCr - pLav * lavCr,
    XNB - pGam * gamNb - pDel * delNb - pLav * lavNb,
    ficGdCr - ficDdCr,
    ficGdNb - ficDdNb,
    ficGdCr - ficLdCr,
    ficGdNb - ficLdNb,
)

ficVars = (gamCr, gamNb, delCr, delNb, lavCr, lavNb)

fictitious = solve(ficEqns, ficVars, dict=True)

# Note: denominator is the determinant, identical by
# definition. So, we separate it to save some FLOPs.
fict_gam_Cr, determinant = fraction(fictitious[0][gamCr])
fict_gam_Nb, determinant = fraction(fictitious[0][gamNb])
fict_del_Cr, determinant = fraction(fictitious[0][delCr])
fict_del_Nb, determinant = fraction(fictitious[0][delNb])
fict_lav_Cr, determinant = fraction(fictitious[0][lavCr])
fict_lav_Nb, determinant = fraction(fictitious[0][lavNb])

inv_fict_det = 1.0 / (factor(expand(gcd * determinant)))

fict_gam_Cr = factor(expand(gcd * fict_gam_Cr)) * INV_DET
fict_gam_Nb = factor(expand(gcd * fict_gam_Nb)) * INV_DET

fict_del_Cr = factor(expand(gcd * fict_del_Cr)) * INV_DET
fict_del_Nb = factor(expand(gcd * fict_del_Nb)) * INV_DET

fict_lav_Cr = factor(expand(gcd * fict_lav_Cr)) * INV_DET
fict_lav_Nb = factor(expand(gcd * fict_lav_Nb)) * INV_DET

# ============ COMPOSITION SHIFTS ============
## Ref: TKR5p219

r_del, r_lav = symbols("r_del, r_lav")

GaCrCr = p_d2Ggam_dxCrCr
GaCrNb = p_d2Ggam_dxCrNb
GaNbNb = p_d2Ggam_dxNbNb

GbCrCr = p_d2Gdel_dxCrCr
GbCrNb = p_d2Gdel_dxCrNb
GbNbNb = p_d2Gdel_dxNbNb

GgCrCr = p_d2Glav_dxCrCr
GgCrNb = p_d2Glav_dxCrNb
GgNbNb = p_d2Glav_dxNbNb

DaCrCr = xe_gam_Cr * GaCrCr
DaCrNb = xe_gam_Cr * GaCrNb
DaNbCr = xe_gam_Nb * GaCrNb
DaNbNb = xe_gam_Nb * GaNbNb

DbCrCr = xe_del_Cr * GbCrCr
DbCrNb = xe_del_Cr * GbCrNb
DbNbCr = xe_del_Nb * GbCrNb
DbNbNb = xe_del_Nb * GbNbNb

DgCrCr = xe_lav_Cr * GgCrCr
DgCrNb = xe_lav_Cr * GgCrNb
DgNbCr = xe_lav_Nb * GgCrNb
DgNbNb = xe_lav_Nb * GgNbNb

# Three-Component Points: Gamma-Delta-Laves Equilibrium with Curvature

A = Matrix(
    [
        [GaCrCr, GaCrNb, -GbCrCr, -GbCrNb, 0, 0],
        [GaCrNb, GaNbNb, -GbCrNb, -GbNbNb, 0, 0],
        [GaCrCr, GaCrNb, 0, 0, -GgCrCr, -GgCrNb],
        [GaCrNb, GaNbNb, 0, 0, -GgCrNb, -GgNbNb],
        [DaCrCr + DaNbCr, DaCrNb + DaNbNb, -DbCrCr - DbNbCr, -DbCrNb - DbNbNb, 0, 0],
        [DaCrCr + DaNbCr, DaCrNb + DaNbNb, 0, 0, -DgCrCr - DgNbCr, -DgCrNb - DgNbNb],
    ]
)

br = Matrix([[0], [0], [0], [0], [-2 * s_delta / r_del], [-2 * s_laves / r_lav]])

xr = A.cholesky_solve(br)

dx_r_gam_Cr = xr[0]
dx_r_gam_Nb = xr[1]
dx_r_del_Cr = xr[2]
dx_r_del_Nb = xr[3]
dx_r_lav_Cr = xr[4]
dx_r_lav_Nb = xr[5]


# === Diffusivity ===

xCr0 = 0.30
xNb0 = 0.02

# *** Sanity Checks ***

print("Free energy at ({0}, {1}):\n".format(xCr0, xNb0))
print("G={0:12g} J/mol".format(G_g.subs({XCR: xCr0, XNB: xNb0})))
print("")

muCrPyC = mu_Cr.subs({XCR: xCr0, XNB: xNb0})
muNbPyC = mu_Nb.subs({XCR: xCr0, XNB: xNb0})
muNiPyC = mu_Ni.subs({XCR: xCr0, XNB: xNb0})
muCrTC = -48499.753
muNbTC = -162987.63
muNiTC = -62416.931
muCrErr = 100 * (muCrTC - muCrPyC) / muCrTC
muNbErr = 100 * (muNbTC - muNbPyC) / muNbTC
muNiErr = 100 * (muNiTC - muNiPyC) / muNiTC

print("Chemical potentials at ({0}, {1}):\n".format(xCr0, xNb0))
print("CR: {0:.3f} J/mol (cf. {1:.3f}: {2:.2f}% error)".format(muCrPyC, muCrTC, muCrErr))
print("NB: {0:.2f} J/mol (cf. {1:.2f}: {2:.2f}% error)".format(muNbPyC, muNbTC, muNbErr))
print("NI: {0:.3f} J/mol (cf. {1:.3f}: {2:.2f}% error)".format(muNiPyC, muNiTC, muNiErr))
print("")


# *** Activation Energies in FCC Ni [J/mol] ***
# Motion of species (1) in pure (2), transcribed from `Ni-Nb-Cr_fcc_mob.tdb`
## Ref: TKR5p286, TKR5p316

Q_Cr_Cr   = -235000 - 82.0 * temp
Q_Cr_Nb   = -287000 - 64.4 * temp
Q_Cr_Ni   = -287000 - 64.4 * temp
Q_Cr_CrNi = -68000

Q_Nb_Cr   = -260955 + RT * log(1.1300e-4)
Q_Nb_Nb   = -102570 + RT * log(1.2200e-5)
Q_Nb_Ni   = -260955 + RT * log(1.1300e-4)
Q_Nb_NbNi = -332498

Q_Ni_Cr   = -235000 - 82.0 * temp
Q_Ni_Nb   = -287060 + RT * log(1.0e-4)
Q_Ni_Ni   = -287000 - 69.8 * temp
Q_Ni_CrNi = -81000
Q_Ni_NbNi = -4207044

Q_Cr = XCR * Q_Cr_Cr + XNB * Q_Cr_Nb + XNI * Q_Cr_Ni + XCR * XNI * Q_Cr_CrNi
Q_Nb = XCR * Q_Nb_Cr + XNB * Q_Nb_Nb + XNI * Q_Nb_Ni + XNB * XNI * Q_Nb_NbNi
Q_Ni = XCR * Q_Ni_Cr + XNB * Q_Ni_Nb + XNI * Q_Ni_Ni + XCR * XNI * Q_Ni_CrNi  + XNB * XNI * Q_Ni_NbNi

# *** Atomic Mobilities in FCC Ni [m²mol/Js due to unit prefactor (m²/s)] ***
## Ref: TKR5p286, TKR5p316

M_Cr = exp(Q_Cr / RT) / RT
M_Nb = exp(Q_Nb / RT) / RT
M_Ni = exp(Q_Ni / RT) / RT

L0_Cr = XCR * M_Cr
L0_Nb = XNB * M_Nb
L0_Ni = XNI * M_Ni

# For DICTRA comparisons ("L0kj") -- Lattice Frame
L0 = Matrix([[L0_Cr, 0, 0],
             [0, L0_Nb, 0],
             [0, 0, L0_Ni]])

print("L0=xM at ({0}, {1}):\n".format(xCr0, xNb0))
pprint(L0.subs({XCR: xCr0, XNB: xNb0}))
print("")

"""
C = Matrix([[duCr_dxCr, duCr_dxNb, duCr_dxNi],
            [duNb_dxCr, duNb_dxNb, duNb_dxNi],
            [duNi_dxCr, duNi_dxNi, duNi_dxNi]])

print("Curvatures at ({0}, {1}):\n".format(xCr0, xNb0))
pprint(C.subs({XCR: xCr0, XNB: xNb0}))
print("")
"""

# *** Diffusivity in FCC Ni [m²/s] ***
## Ref: TKR5p333 (Campbell's method)

D_CrCr = (1 - XCR) * L0_Cr * duCr_dxCr      - XCR  * L0_Nb * duNb_dxCr      - XCR  * L0_Ni * duNi_dxCr
D_CrNb = (1 - XCR) * L0_Cr * duCr_dxNb      - XCR  * L0_Nb * duNb_dxNb      - XCR  * L0_Ni * duNi_dxNb
D_CrNi = (1 - XCR) * L0_Cr * duCr_dxNi      - XCR  * L0_Nb * duNb_dxNi      - XCR  * L0_Ni * duNi_dxNi

D_NbCr =    - XNB  * L0_Cr * duCr_dxCr + (1 - XNB) * L0_Nb * duNb_dxCr      - XNB  * L0_Ni * duNi_dxCr
D_NbNb =    - XNB  * L0_Cr * duCr_dxNb + (1 - XNB) * L0_Nb * duNb_dxNb      - XNB  * L0_Ni * duNi_dxNb
D_NbNi =    - XNB  * L0_Cr * duCr_dxNi + (1 - XNB) * L0_Nb * duNb_dxNi      - XNB  * L0_Ni * duNi_dxNi

D_NiCr =    - XNI  * L0_Cr * duCr_dxCr      - XNI  * L0_Nb * duNb_dxCr + (1 - XNI) * L0_Ni * duNi_dxCr
D_NiNb =    - XNI  * L0_Cr * duCr_dxNb      - XNI  * L0_Nb * duNb_dxNb + (1 - XNI) * L0_Ni * duNi_dxNb
D_NiNi =    - XNI  * L0_Cr * duCr_dxNi      - XNI  * L0_Nb * duNb_dxNi + (1 - XNI) * L0_Ni * duNi_dxNi

DCC = D_CrCr.subs({XCR: xCr0, XNB: xNb0})
DCN = D_CrNb.subs({XCR: xCr0, XNB: xNb0})
DCn = D_CrNi.subs({XCR: xCr0, XNB: xNb0})

DNC = D_NbCr.subs({XCR: xCr0, XNB: xNb0})
DNN = D_NbNb.subs({XCR: xCr0, XNB: xNb0})
DNn = D_NbNi.subs({XCR: xCr0, XNB: xNb0})

DnC = D_NiCr.subs({XCR: xCr0, XNB: xNb0})
DnN = D_NiNb.subs({XCR: xCr0, XNB: xNb0})
Dnn = D_NiNi.subs({XCR: xCr0, XNB: xNb0})

print("Diffusivity at ({0}, {1}):\n".format(xCr0, xNb0))
print("⎡{0:13.3g} {1:13.3g} {2:13.3g}⎤".format(DCC, DCN, DCn))
print("⎢                                         ⎥")
print("⎢{0:13.3g} {1:13.3g} {2:13.3g}⎥".format(DNC, DNN, DNn))
print("⎢                                         ⎥")
print("⎣{0:13.3g} {1:13.3g} {2:13.3g}⎦".format(DnC, DnN, Dnn))
print("")

# From C. Campbell, 2020-03-10 via Thermo-Calc
TC_CrCr = 2.85167e-17
TC_CrNb = 1.16207e-17
TC_CrNi = 4.74808e-18
TC_NbCr = -4.28244e-18
TC_NbNb = 1.56459e-16
TC_NbNi = -3.88918e-17
TC_NiCr = -2.42343e-17
TC_NiNb = -1.68080e-16
TC_NiNi = 3.41437e-17

print("Reference Diffusivity at ({0}, {1}):\n".format(xCr0, xNb0))
print("⎡{0:13.3g} {1:13.3g} {2:13.3g}⎤".format(TC_CrCr, TC_CrNb, TC_CrNi))
print("⎢                                         ⎥")
print("⎢{0:13.3g} {1:13.3g} {2:13.3g}⎥".format(TC_NbCr, TC_NbNb, TC_NbNi))
print("⎢                                         ⎥")
print("⎣{0:13.3g} {1:13.3g} {2:13.3g}⎦".format(TC_NiCr, TC_NiNb, TC_NiNi))
print("")

DE_CrCr = 100 * (DCC - TC_CrCr) / TC_CrCr
DE_CrNb = 100 * (DCN - TC_CrNb) / TC_CrNb
DE_CrNi = 100 * (DCn - TC_CrNi) / TC_CrNb
DE_NbCr = 100 * (DNC - TC_NbCr) / TC_NbCr
DE_NbNb = 100 * (DNN - TC_NbNb) / TC_NbNb
DE_NbNi = 100 * (DNn - TC_NbNi) / TC_NbNi
DE_NiCr = 100 * (DnC - TC_NiCr) / TC_NiCr
DE_NiNb = 100 * (DnN - TC_NiNb) / TC_NiNb
DE_NiNi = 100 * (Dnn - TC_NiNi) / TC_NiNi

print("Diffusivity Error at ({0}, {1}):\n".format(xCr0, xNb0))
print("⎡{0:10.2f}% {1:10.2f}% {2:10.2f}%⎤".format(DE_CrCr, DE_CrNb, DE_CrNi))
print("⎢                                   ⎥")
print("⎢{0:10.2f}% {1:10.2f}% {2:10.2f}%⎥".format(DE_NbCr, DE_NbNb, DE_NbNi))
print("⎢                                   ⎥")
print("⎣{0:10.2f}% {1:10.2f}% {2:10.2f}%⎦".format(DE_NiCr, DE_NiNb, DE_NiNi))
print("")


"""
# $D^n_{kj} = D_{kj} - D_{kn}$

Dr =  Matrix([[D_CrCr - D_CrNi, D_CrNb - D_CrNi],
              [D_NbCr - D_NbNi, D_NbNb - D_NbNi],
              [D_NiCr - D_NiNi, D_NiNb - D_NiNi]])

print("Reduced Diffusivity at ({0}, {1}):\n".format(xCr0, xNb0))
pprint(Dr.subs({XCR: xCr0, XNB: xNb0}))
print("")
"""

# === Export C source code ===

codegen(
    [  # Interpolator
        ("p", interpolator),
        ("pPrime", dinterpdx),
        ("interface_profile", interfaceProfile),
        # temperature
        ("kT", 1.380649e-23 * temp),
        ("RT", 8.314468 * temp),
        ("Vm", Vm),
        # Equilibrium Compositions
        ("xe_gam_Cr", xe_gam_Cr),
        ("xe_gam_Nb", xe_gam_Nb),
        ("xe_del_Cr", xe_del_Cr),
        ("xe_del_Nb", xe_del_Nb),
        ("xe_lav_Cr", xe_lav_Cr),
        ("xe_lav_Nb", xe_lav_Nb),
        # Matrix composition range
        ("matrix_min_Cr", matrixMinCr),
        ("matrix_max_Cr", matrixMaxCr),
        ("matrix_min_Nb", matrixMinNb),
        ("matrix_max_Nb", matrixMaxNb),
        # Enriched composition range
        ("enrich_min_Cr", enrichMinCr),
        ("enrich_max_Cr", enrichMaxCr),
        ("enrich_min_Nb", enrichMinNb),
        ("enrich_max_Nb", enrichMaxNb),
        # Curvature-Corrected Compositions
        ("xr_gam_Cr", xe_gam_Cr + dx_r_gam_Cr),
        ("xr_gam_Nb", xe_gam_Nb + dx_r_gam_Nb),
        ("xr_del_Cr", xe_del_Cr + dx_r_del_Cr),
        ("xr_del_Nb", xe_del_Nb + dx_r_del_Nb),
        ("xr_lav_Cr", xe_lav_Cr + dx_r_lav_Cr),
        ("xr_lav_Nb", xe_lav_Nb + dx_r_lav_Nb),
        # Fictitious compositions
        ("inv_fict_det", inv_fict_det),
        ("fict_gam_Cr", fict_gam_Cr),
        ("fict_gam_Nb", fict_gam_Nb),
        ("fict_del_Cr", fict_del_Cr),
        ("fict_del_Nb", fict_del_Nb),
        ("fict_lav_Cr", fict_lav_Cr),
        ("fict_lav_Nb", fict_lav_Nb),
        # Interfacial energies
        ("s_delta", s_delta),
        ("s_laves", s_laves),
        # Gibbs energies
        ("CALPHAD_gam", g_gamma),
        ("CALPHAD_del", g_delta),
        ("CALPHAD_lav", g_laves),
        ("g_gam", p_gamma),
        ("g_del", p_delta),
        ("g_lav", p_laves),
        # First derivatives
        ("dg_gam_dxCr", p_dGgam_dxCr),
        ("dg_gam_dxNb", p_dGgam_dxNb),
        ("dg_del_dxCr", p_dGdel_dxCr),
        ("dg_del_dxNb", p_dGdel_dxNb),
        ("dg_lav_dxCr", p_dGlav_dxCr),
        ("dg_lav_dxNb", p_dGlav_dxNb),
        # Second derivatives
        ("d2g_gam_dxCrCr", p_d2Ggam_dxCrCr),
        ("d2g_gam_dxCrNb", p_d2Ggam_dxCrNb),
        ("d2g_gam_dxNbCr", p_d2Ggam_dxNbCr),
        ("d2g_gam_dxNbNb", p_d2Ggam_dxNbNb),
        ("d2g_del_dxCrCr", p_d2Gdel_dxCrCr),
        ("d2g_del_dxCrNb", p_d2Gdel_dxCrNb),
        ("d2g_del_dxNbCr", p_d2Gdel_dxNbCr),
        ("d2g_del_dxNbNb", p_d2Gdel_dxNbNb),
        ("d2g_lav_dxCrCr", p_d2Glav_dxCrCr),
        ("d2g_lav_dxCrNb", p_d2Glav_dxCrNb),
        ("d2g_lav_dxNbCr", p_d2Glav_dxNbCr),
        ("d2g_lav_dxNbNb", p_d2Glav_dxNbNb),
        # Mobilities
        ("L0_Cr", L0_Cr),
        ("L0_Nb", L0_Nb),
        ("L0_Ni", L0_Ni),
        # Diffusivities
        ("D_CrCr", D_CrCr - D_CrNi), ("D_CrNb", D_CrNb - D_CrNi),
        ("D_NbCr", D_NbCr - D_NbNi), ("D_NbNb", D_NbNb - D_NbNi)
    ],
    language="C",
    prefix="parabola625",
    project="PrecipitateAging",
    to_files=True,
)
