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
from sympy import diff, Eq, expand, factor, fraction, Matrix, symbols
from sympy.abc import x, r, y, z, L
from sympy.core.numbers import pi
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.trigonometric import tanh
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers import solve, solve_linear_system
from sympy.utilities.codegen import codegen

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
XCR, XNB = symbols("XCR XNB")

# Define lever rule equations
# Ref: TKR4p161, 172; TKR5p266, 272, 293
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


# triangle bounding three-phase coexistence
X0 = [
    simX(xe_gam_Nb, xe_gam_Cr),
    simX(xe_del_Nb, xe_del_Cr),
    simX(xe_lav_Nb, xe_lav_Cr),
    simX(xe_gam_Nb, xe_gam_Cr),
]
Y0 = [simY(xe_gam_Cr), simY(xe_del_Cr), simY(xe_lav_Cr), simY(xe_gam_Cr)]

# Make sublattice -> system substitutions
g_gamma = inVm * g_gamma.subs(
    {
        symbols("FCC_A10CR"): XCR,
        symbols("FCC_A10NB"): XNB,
        symbols("FCC_A10NI"): 1 - XCR - XNB,
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

# Generate paraboloid expressions (2nd-order Taylor series approximations)

# Curvatures
PC_gam_CrCr = diff(g_gamma, XCR, XCR).subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})
PC_gam_CrNb = diff(g_gamma, XCR, XNB).subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})
PC_gam_NbNb = diff(g_gamma, XNB, XNB).subs({XCR: xe_gam_Cr, XNB: xe_gam_Nb})

PC_del_CrCr = diff(g_delta, XCR, XCR).subs({XCR: xe_del_Cr, XNB: xe_del_Nb})
PC_del_CrNb = diff(g_delta, XCR, XNB).subs({XCR: xe_del_Cr, XNB: xe_del_Nb})
PC_del_NbNb = diff(g_delta, XNB, XNB).subs({XCR: xe_del_Cr, XNB: xe_del_Nb})

PC_lav_CrCr = diff(g_laves, XCR, XCR).subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})
PC_lav_CrNb = diff(g_laves, XCR, XNB).subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})
PC_lav_NbNb = diff(g_laves, XNB, XNB).subs({XCR: xe_lav_Cr, XNB: xe_lav_Nb})

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

fict_gam_Cr, det_gam_Cr = fraction(fictitious[0][gamCr])
fict_gam_Nb, det_gam_Nb = fraction(fictitious[0][gamNb])
fict_del_Cr, det_del_Cr = fraction(fictitious[0][delCr])
fict_del_Nb, det_del_Nb = fraction(fictitious[0][delNb])
fict_lav_Cr, det_lav_Cr = fraction(fictitious[0][lavCr])
fict_lav_Nb, det_lav_Nb = fraction(fictitious[0][lavNb])

inv_fict_det = 1.0 / (factor(expand(gcd * det_gam_Cr)))

fict_gam_Cr = factor(expand(gcd * fict_gam_Cr)) * INV_DET
fict_gam_Nb = factor(expand(gcd * fict_gam_Nb)) * INV_DET

fict_del_Cr = factor(expand(gcd * fict_del_Cr)) * INV_DET
fict_del_Nb = factor(expand(gcd * fict_del_Nb)) * INV_DET

fict_lav_Cr = factor(expand(gcd * fict_lav_Cr)) * INV_DET
fict_lav_Nb = factor(expand(gcd * fict_lav_Nb)) * INV_DET

# ============ COMPOSITION SHIFTS ============
# Ref: TKR5p219

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

# === Atomic Mobilities in FCC Ni ===
# Mobility of (1) in pure (2), from `NIST-nifcc-mob.TDB`
## TKR5p286

M_Cr_Cr   = exp((-235000 - 82.0 * temp) / RT) / RT
M_Cr_Nb   = exp((-287000 - 64.4 * temp) / RT) / RT
M_Cr_Ni   = exp((-287000 - 64.4 * temp) / RT) / RT
M_Cr_CrNi = exp((-68000)                / RT) / RT

M_Nb_Cr   = exp((-255333 + RT * log(7.6071E-5)) / RT) / RT
M_Nb_Nb   = exp((-274328 + RT * log(8.6440E-5)) / RT) / RT
M_Nb_Ni   = exp((-255333 + RT * log(7.6071e-5)) / RT) / RT

M_Ni_Cr   = exp((-235000 - 82.0 * temp)      / RT) / RT
M_Ni_Nb   = exp((-287000 + RT * log(1.0E-4)) / RT) / RT
M_Ni_Ni   = exp((-287000 - 69.8 * temp)      / RT) / RT
M_Ni_CrNi = exp((-81000)                     / RT) / RT

M_Cr = XCR * M_Cr_Cr + XNB * M_Cr_Nb + (1 - XCR - XNB) * M_Cr_Ni + XCR * (1 - XCR - XNB) * M_Cr_CrNi
M_Nb = XCR * M_Nb_Cr + XNB * M_Nb_Nb + (1 - XCR - XNB) * M_Nb_Ni
M_Ni = XCR * M_Ni_Cr + XNB * M_Ni_Nb + (1 - XCR - XNB) * M_Ni_Ni + XCR * (1 - XCR - XNB) * M_Ni_CrNi

# === Chemical Mobilities in FCC Ni ===
## TKR5p292 and 311-312

phi_del, phi_lav = symbols("phi_del, phi_lav")

M_CrCr = Vm**3 * ( M_Cr * (1 - XCR)**2    + M_Nb * XCR**2          + M_Ni * XCR**2)
M_CrNb = Vm**3 * (-M_Cr * (1 - XCR) * XNB - M_Nb * XCR * (1 - XNB) + M_Ni * XCR * XNB)
M_NbCr = Vm**3 * (-M_Cr * (1 - XCR) * XNB - M_Nb * XCR * (1 - XNB) + M_Ni * XCR * XNB)
M_NbNb = Vm**3 * ( M_Cr * XNB**2          + M_Nb * (1 - XNB)**2    + M_Ni * XNB**2)

# Generate numerically efficient C-code

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
        ("M_CrCr", M_CrCr), ("M_CrNb", M_CrNb),
        ("M_NbCr", M_NbCr), ("M_NbNb", M_NbNb)
    ],
    language="C",
    prefix="parabola625",
    project="PrecipitateAging",
    to_files=True,
)
