# -*- coding: utf-8 -*-

# Numerical and CAS libraries
import numpy as np
import os
from sympy import diff, expand, factor, symbols, tanh
from sympy.abc import h, w, x
from sympy.solvers import solve
from sympy.utilities import lambdify
import sys

# I/O libraries
from sympy.parsing.sympy_parser import parse_expr

# Thermodynamics libraries
from pycalphad import Database, Model, calculate

# Global variables and shared functions
sys.path.append(os.path.join(os.path.dirname(__file__), "../thermo"))
from constants import *

# Phase-field interpolator
p = x**3 * (6. * x**2 - 15. * x + 10.)
p_prime = p.diff(x)

# Interface interpolator
smooth_interface = 0.5 * (1 - tanh((x - h) / (2 * w)))

# Declare sublattice variables used in Pycalphad expressions
XCR, XNB = symbols("XCR XNB")
XNI = 1 - XCR - XNB
XEQ = {XCR: xe_gam_Cr, XNB: xe_gam_Nb}

# Read CALPHAD database from disk, specify phases and elements of interest
tdb = Database("../thermo/Du_Cr-Nb-Ni_simple.tdb")
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
        symbols("D0A_NBNI31CR"): XCR / 0.75,
        symbols("D0A_NBNI31NI"): 1 - XCR / 0.75,
        symbols("T"): temp,
    }
)

g_laves = inVm * g_laves.subs(
    {
        symbols("C14_LAVES0CR"): 1 - 1.5 * XNI,
        symbols("C14_LAVES0NI"): 1.5 * XNI,
        symbols("C14_LAVES1CR"): 1 - 3 * XNB,
        symbols("C14_LAVES1NB"): 3 * XNB,
        symbols("T"): temp,
    }
)

# Generate paraboloid expressions (2nd-order Taylor series approximations)

# Curvatures
PC_gam_CrCr = g_gamma.diff(XCR, XCR).subs(XEQ)
PC_gam_CrNb = g_gamma.diff(XCR, XNB).subs(XEQ)
PC_gam_NbNb = g_gamma.diff(XNB, XNB).subs(XEQ)

PC_del_CrCr = g_delta.diff(XCR, XCR).subs(XEQ)
PC_del_CrNb = g_delta.diff(XCR, XNB).subs(XEQ)
PC_del_NbNb = g_delta.diff(XNB, XNB).subs(XEQ)

PC_lav_CrCr = g_laves.diff(XCR, XCR).subs(XEQ)
PC_lav_CrNb = g_laves.diff(XCR, XNB).subs(XEQ)
PC_lav_NbNb = g_laves.diff(XNB, XNB).subs(XEQ)

# Expressions
### Warning: CALPHAD expressions will be lost! ###
g_gamma = (
    0.5 * PC_gam_CrCr * (XCR - xe_gam_Cr) ** 2
    + PC_gam_CrNb * (XCR - xe_gam_Cr) * (XNB - xe_gam_Nb)
    + 0.5 * PC_gam_NbNb * (XNB - xe_gam_Nb) ** 2
)

g_delta = (
    0.5 * PC_del_CrCr * (XCR - xe_del_Cr) ** 2
    + PC_del_CrNb * (XCR - xe_del_Cr) * (XNB - xe_del_Nb)
    + 0.5 * PC_del_NbNb * (XNB - xe_del_Nb) ** 2
)

g_laves = (
    0.5 * PC_lav_CrCr * (XCR - xe_lav_Cr) ** 2
    + PC_lav_CrNb * (XCR - xe_lav_Cr) * (XNB - xe_lav_Nb)
    + 0.5 * PC_lav_NbNb * (XNB - xe_lav_Nb) ** 2
)

# Generate first derivatives of paraboloid landscape
dGgam_dxCr = diff(g_gamma, XCR)
dGgam_dxNb = diff(g_gamma, XNB)

dGdel_dxCr = diff(g_delta, XCR)
dGdel_dxNb = diff(g_delta, XNB)

dGlav_dxCr = diff(g_laves, XCR)
dGlav_dxNb = diff(g_laves, XNB)

# Generate second derivatives of paraboloid landscape
d2Ggam_dxCrCr = diff(g_gamma, XCR, XCR)
d2Ggam_dxCrNb = diff(g_gamma, XCR, XNB)
d2Ggam_dxNbCr = diff(g_gamma, XNB, XCR)
d2Ggam_dxNbNb = diff(g_gamma, XNB, XNB)

d2Gdel_dxCrCr = diff(g_delta, XCR, XCR)
d2Gdel_dxCrNb = diff(g_delta, XCR, XNB)
d2Gdel_dxNbCr = diff(g_delta, XNB, XCR)
d2Gdel_dxNbNb = diff(g_delta, XNB, XNB)

d2Glav_dxCrCr = diff(g_laves, XCR, XCR)
d2Glav_dxCrNb = diff(g_laves, XCR, XNB)
d2Glav_dxNbCr = diff(g_laves, XNB, XCR)
d2Glav_dxNbNb = diff(g_laves, XNB, XNB)

# Chemical potentials of paraboloid landscape
mu_Cr = Vm * (g_gamma      - XNB  * dGgam_dxNb + (1 - XCR) * dGgam_dxCr)
mu_Nb = Vm * (g_gamma + (1 - XNB) * dGgam_dxNb      - XCR  * dGgam_dxCr)
mu_Ni = Vm * (g_gamma      - XNB  * dGgam_dxNb      - XCR  * dGgam_dxCr)

# Derivatives of μ from SymPy
## N.B.: Since Ni is dependent, duX_dxNi ≡ 0
duCr_dxCr = diff(mu_Cr, XCR)
duCr_dxNb = diff(mu_Cr, XNB)

duNb_dxCr = diff(mu_Nb, XCR)
duNb_dxNb = diff(mu_Nb, XNB)

duNi_dxCr = diff(mu_Ni, XCR)
duNi_dxNb = diff(mu_Ni, XNB)

# ========= FICTITIOUS COMPOSITIONS ==========
# Derivation: TKR4p181
gamCr, gamNb = symbols("gamCr, gamNb")
delCr, delNb = symbols("delCr, delNb")
lavCr, lavNb = symbols("lavCr, lavNb")
pGam, pDel, pLav = symbols("pGam, pDel, pLav")

ficGdCr = dGgam_dxCr.subs({XCR: gamCr, XNB: gamNb})
ficGdNb = dGgam_dxNb.subs({XCR: gamCr, XNB: gamNb})
ficDdCr = dGdel_dxCr.subs({XCR: delCr, XNB: delNb})
ficDdNb = dGdel_dxNb.subs({XCR: delCr, XNB: delNb})
ficLdCr = dGlav_dxCr.subs({XCR: lavCr, XNB: lavNb})
ficLdNb = dGlav_dxNb.subs({XCR: lavCr, XNB: lavNb})

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

# Lambdify all the things!
module="numpy"

p = lambdify(x, p, modules=module)
p_prime = lambdify(x, p_prime, modules=module)
smooth_interface = lambdify((h, w, x), smooth_interface, modules=module)

g_gamma = lambdify((XCR, XNB), g_gamma, modules=module)
g_delta = lambdify((XCR, XNB), g_delta, modules=module)

dg_gam_dxCr = lambdify((XCR, XNB), dGgam_dxCr, modules=module)
dg_gam_dxNb = lambdify((XCR, XNB), dGgam_dxNb, modules=module)
dg_del_dxCr = lambdify((XCR, XNB), dGdel_dxCr, modules=module)
dg_del_dxNb = lambdify((XCR, XNB), dGdel_dxNb, modules=module)

d2g_gam_dxCrCr = lambdify((), d2Ggam_dxCrCr, modules=module)
d2g_gam_dxCrNb = lambdify((), d2Ggam_dxCrNb, modules=module)
d2g_gam_dxNbCr = lambdify((), d2Ggam_dxNbCr, modules=module)
d2g_gam_dxNbNb = lambdify((), d2Ggam_dxNbNb, modules=module)
d2g_del_dxCrCr = lambdify((), d2Gdel_dxCrCr, modules=module)
d2g_del_dxCrNb = lambdify((), d2Gdel_dxCrNb, modules=module)
d2g_del_dxNbCr = lambdify((), d2Gdel_dxNbCr, modules=module)
d2g_del_dxNbNb = lambdify((), d2Gdel_dxNbNb, modules=module)

ficVars = (XCR, XNB, pDel, pGam, pLav)

x_gam_Cr = lambdify(ficVars, factor(expand(fictitious[0][gamCr])), modules=module)
x_gam_Nb = lambdify(ficVars, factor(expand(fictitious[0][gamNb])), modules=module)
x_del_Cr = lambdify(ficVars, factor(expand(fictitious[0][delCr])), modules=module)
x_del_Nb = lambdify(ficVars, factor(expand(fictitious[0][delNb])), modules=module)

if __name__ == "__main__":
    from sympy import init_printing, pprint
    init_printing()
    pprint(factor(expand(fictitious[0][gamCr])))
    pprint(factor(expand(fictitious[0][gamNb])))
