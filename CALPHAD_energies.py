#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################################################################
# This software was developed at the National Institute of Standards and Technology #
# by employees of the Federal Government in the course of their official duties.    #
# Pursuant to title 17 section 105 of the United States Code this software is not   #
# subject to copyright protection and is in the public domain. NIST assumes no      #
# responsibility whatsoever for the use of this code by other parties, and makes no #
# guarantees, expressed or implied, about its quality, reliability, or any other    #
# characteristic. We would appreciate acknowledgement if the software is used.      #
#                                                                                   #
# This software can be redistributed and/or modified freely provided that any       #
# derivative works bear some notice that they are derived from it, and any modified #
# versions bear some notice that they have been modified.                           #
#####################################################################################

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
# - δ: eliminate Nb from the second (Ni) sublattice, $\mathrm{(\mathbf{Nb}, Ni)_1(Cr, \mathbf{Ni})_3}$
#      * $y_\mathrm{Nb}'  = 4x_\mathrm{Nb}$
#      * $y_\mathrm{Ni}'  = 1 - 4x_\mathrm{Nb}$
#      * $y_\mathrm{Cr}'' = \frac{4}{3}x_\mathrm{Cr}$
#      * $y_\mathrm{Ni}'' = 1 - \frac{4}{3}x_\mathrm{Cr}$
#      * Constraints: $x_\mathrm{Nb}\leq\frac{1}{4}$
#                     $x_\mathrm{Cr}\leq\frac{3}{4}$
#
# - Laves: eliminate Nb from the first (Cr) sublattice, $\mathrm{(\mathbf{Cr}, Ni)_2(Cr, \mathbf{Nb})_1}$
#      * $y_\mathrm{Cr}'  = 1 - \frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Ni}'  = \frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Cr}'' = 1 - 3x_\mathrm{Nb}$
#      * $y_\mathrm{Nb}'' = 3x_\mathrm{Nb}$
#      * Constraints: $0\leq x_\mathrm{Ni}\leq\frac{2}{3}$
#                     $0\leq x_\mathrm{Nb}\leq\frac{1}{3}$

# Numerical libraries
import numpy as np
from sympy.utilities.lambdify import lambdify

# Thermodynamics and computer-algebra libraries
from pycalphad import Database, calculate, Model
from sympy.utilities.codegen import codegen
from sympy.parsing.sympy_parser import parse_expr
from sympy import And, Ge, Gt, Le, Lt, Or, Piecewise, true
from sympy import diff, Function, Lambda, symbols, simplify, sympify
from sympy import Abs, exp, log, pi, tanh

# Constants
epsilon = 1e-10 # tolerance for comparing floating-point numbers to zero
temp = 870 + 273.15 # 1143 Kelvin

RT = 8.3144598*temp # J/mol/K
Vm = 1.0e-5         # m^3/mol
inVm = 1.0 / Vm     # mol/m^3

# Let's avoid integer arithmetic in fractions.
fr3by4 = 0.75
fr3by2 = 1.5
fr4by3 = 4.0/3
fr2by3 = 2.0/3
fr1by4 = 0.25
fr1by3 = 1.0/3
fr1by2 = 0.5
rt3by2 = np.sqrt(3.0)/2

# Helper functions to convert compositions into (x,y) coordinates
def simX(x2, x3):
    return x2 + fr1by2 * x3

def simY(x3):
    return rt3by2 * x3

# triangle bounding the Gibbs simplex
XS = [0, simX(1,0), simX(0,1), 0]
YS = [0, simY(0),   simY(1),   0]

# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    xnb = -0.002
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.002 - xcr
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))
    # Nb-Ni edge
    xcr = -0.002
    xnb = 0.05*i
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))

# Read CALPHAD database from disk, specify phases and elements of interest
tdb = Database('Du_Cr-Nb-Ni_simple.tdb')
elements = ['CR', 'NB', 'NI']

species = list(set([i for c in tdb.phases['FCC_A1'].constituents for i in c]))
model = Model(tdb, species, 'FCC_A1')
g_gamma = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases['D0A_NBNI3'].constituents for i in c]))
model = Model(tdb, species, 'D0A_NBNI3')
g_delta = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases['C14_LAVES'].constituents for i in c]))
model = Model(tdb, species, 'C14_LAVES')
g_laves = parse_expr(str(model.ast))


# Declare sublattice variables used in Pycalphad expressions
XCR, XNB, XNI = symbols('XCR XNB XNI')
T = symbols('T')

# Gamma
FCC_A10CR, FCC_A10NB, FCC_A10NI, FCC_A11VA = symbols('FCC_A10CR FCC_A10NB FCC_A10NI FCC_A11VA')

# Delta
D0A_NBNI30NI, D0A_NBNI30NB, D0A_NBNI31CR, D0A_NBNI31NI = symbols('D0A_NBNI30NI D0A_NBNI30NB D0A_NBNI31CR D0A_NBNI31NI')

# Laves
C14_LAVES0CR, C14_LAVES0NI, C14_LAVES1CR, C14_LAVES1NB = symbols('C14_LAVES0CR C14_LAVES0NI C14_LAVES1CR C14_LAVES1NB')

# Specify gamma-delta-Laves corners (from phase diagram)
xe_gam_Cr = 0.490
xe_gam_Nb = 0.025
xe_gam_Ni = 1 - xe_gam_Cr - xe_gam_Nb

xe_del_Cr = 0.015
xe_del_Nb = 0.245

xe_lav_Cr = 0.300
xe_lav_Nb = 0.328
xe_lav_Ni = 1 - xe_lav_Cr - xe_lav_Nb

# Specify Taylor series expansion points
xt_gam_Cr = 0.400
xt_gam_Nb = 0.200
xt_gam_Ni = 1 - xt_gam_Cr - xt_gam_Nb

xt_del_Cr = 0.100
xt_del_Nb = 0.245

xt_lav_Cr = 0.350
xt_lav_Nb = 0.200
xt_lav_Ni = 1 - xt_lav_Cr - xt_lav_Nb

# Specify upper limit compositions
xcr_del_hi = fr3by4
xnb_del_hi = fr1by4

xnb_lav_hi = fr1by3
xni_lav_hi = fr2by3
xni_lav_hi = fr2by3


# Anchor points for Taylor series
XT = [simX(xt_gam_Nb, xt_gam_Cr), simX(xt_del_Nb, xt_del_Cr), simX(xt_lav_Nb, xt_lav_Cr)]
YT = [simY(xt_gam_Cr),            simY(xt_del_Cr),            simY(xt_lav_Cr)]

# triangle bounding three-phase coexistence
X0 = [simX(xe_gam_Nb, xe_gam_Cr), simX(xe_del_Nb, xe_del_Cr), simX(xe_lav_Nb, xe_lav_Cr)]
Y0 = [simY(xe_gam_Cr),            simY(xe_del_Cr),            simY(xe_lav_Cr)]

# Make sublattice -> system substitutions
g_gamma = inVm * g_gamma.subs({FCC_A10CR: XCR,
                               FCC_A10NB: XNB,
                               FCC_A10NI: 1 - XCR - XNB,
                               FCC_A11VA: 1,
                               T: temp})

g_delta = inVm * g_delta.subs({D0A_NBNI30NB: 4*XNB,
                               D0A_NBNI30NI: 1 - 4*XNB,
                               D0A_NBNI31CR: fr4by3 * XCR,
                               D0A_NBNI31NI: 1 - fr4by3 * XCR,
                               T: temp})

g_laves = inVm * g_laves.subs({C14_LAVES0CR: 1 - fr3by2 * (1 - XCR - XNB),
                               C14_LAVES0NI: fr3by2 * (1 - XCR - XNB),
                               C14_LAVES1CR: 1 - 3*XNB,
                               C14_LAVES1NB: 3 * XNB,
                               T: temp})

# Generate parabolic expressions (the crudest of approximations)

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
p_gamma = fr1by2 * PC_gam_CrCr * (XCR - xe_gam_Cr)**2                      \
        +          PC_gam_CrNb * (XCR - xe_gam_Cr)    * (XNB - xe_gam_Nb)  \
        + fr1by2 * PC_gam_NbNb                        * (XNB - xe_gam_Nb)**2

# print("Gamma:\n d2f/dcr2 = {0}\n d2f/dnb2 = {1}\n d2f/dcr.dnb = {2}".format(PC_gam_CrCr, PC_gam_NbNb, PC_gam_CrNb))

p_delta = fr1by2 * PC_del_CrCr * (XCR - xe_del_Cr)**2                      \
        +          PC_del_CrNb * (XCR - xe_del_Cr)    * (XNB - xe_del_Nb)  \
        + fr1by2 * PC_del_NbNb                        * (XNB - xe_del_Nb)**2

# print("Delta:\n d2f/dcr2 = {0}\n d2f/dnb2 = {1}\n d2f/dcr.dnb = {2}".format(PC_del_CrCr, PC_del_NbNb, PC_del_CrNb))

p_laves = fr1by2 * PC_lav_CrCr * (XCR - xe_lav_Cr)**2                      \
        +          PC_lav_CrNb * (XCR - xe_lav_Cr)    * (XNB - xe_lav_Nb)  \
        + fr1by2 * PC_lav_NbNb                        * (XNB - xe_lav_Nb)**2

# print("Laves:\n d2f/dcr2 = {0}\n d2f/dnb2 = {1}\n d2f/dcr.dnb = {2}".format(PC_lav_CrCr, PC_lav_NbNb, PC_lav_CrNb))

# Generate first derivatives of Taylor series landscape
p_dGgam_dxCr = diff(p_gamma, XCR)
p_dGgam_dxNb = diff(p_gamma, XNB)

p_dGdel_dxCr = diff(p_delta, XCR)
p_dGdel_dxNb = diff(p_delta, XNB)

p_dGlav_dxCr = diff(p_laves, XCR)
p_dGlav_dxNb = diff(p_laves, XNB)

# Generate second derivatives of Taylor series landscape
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

# Write parabolic functions as C code
codegen([# Gibbs energies
         ('g_gam', p_gamma),         ('g_del', p_delta),         ('g_lav', p_laves),
         # Constants
         ('xe_gam_Cr', xe_gam_Cr),         ('xe_gam_Nb', xe_gam_Nb),
         ('xe_del_Cr', xe_del_Cr),         ('xe_del_Nb', xe_del_Nb),
         ('xe_lav_Cr', xe_lav_Cr),         ('xe_lav_Nb', xe_lav_Nb),
         # First derivatives
         ('dg_gam_dxCr', p_dGgam_dxCr),         ('dg_gam_dxNb', p_dGgam_dxNb),
         ('dg_del_dxCr', p_dGdel_dxCr),         ('dg_del_dxNb', p_dGdel_dxNb),
         ('dg_lav_dxCr', p_dGlav_dxCr),         ('dg_lav_dxNb', p_dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr', p_d2Ggam_dxCrCr),         ('d2g_gam_dxCrNb', p_d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr', p_d2Ggam_dxNbCr),         ('d2g_gam_dxNbNb', p_d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr', p_d2Gdel_dxCrCr),         ('d2g_del_dxCrNb', p_d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr', p_d2Gdel_dxNbCr),         ('d2g_del_dxNbNb', p_d2Gdel_dxNbNb),
         ('d2g_lav_dxCrCr', p_d2Glav_dxCrCr),         ('d2g_lav_dxCrNb', p_d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr', p_d2Glav_dxNbCr),         ('d2g_lav_dxNbNb', p_d2Glav_dxNbNb)],
        language='C', prefix='parabola625', project='ALLOY625', to_files=True)

# Generate numerically efficient system-composition expressions

# Lambdify parabolic expressions
PG = lambdify([XCR, XNB], p_gamma, modules='sympy')
PD = lambdify([XCR, XNB], p_delta, modules='sympy')
PL = lambdify([XCR, XNB], p_laves, modules='sympy')
