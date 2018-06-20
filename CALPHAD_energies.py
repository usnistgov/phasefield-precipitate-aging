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
from sympy import diff, Function, Lambda, Matrix, symbols, simplify, sympify
from sympy import Abs, exp, log, pi, tanh, solve_linear_system

# Constants
from constants import *

# Read CALPHAD database from disk, specify phases and elements of interest
tdb = Database('thermo/Du_Cr-Nb-Ni_simple.tdb')
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

xe_del_Cr = 0.015
xe_del_Nb = 0.245

xe_lav_Cr = 0.300
xe_lav_Nb = 0.328

# Define lever rule equations
xo, yo = symbols('xo yo')
xb, yb = symbols('xb yb')
xc, yc = symbols('xc yc')
xd, yd = symbols('xd yd')

from sympy.abc import x, y
levers = solve_linear_system(Matrix( ((yo - yb, xb - xo, xb * yo - xo * yb),
                                      (yc - yd, xd - xc, xd * yc - xc * yd)) ), x, y)
def draw_bisector(A, B):
    bNb = (A * xe_del_Nb + B * xe_lav_Nb) / (A + B)
    bCr = (A * xe_del_Cr + B * xe_lav_Cr) / (A + B)
    x = [simX(xe_gam_Nb, xe_gam_Cr), simX(bNb, bCr)]
    y = [simY(xe_gam_Cr), simY(bCr)]
    return x, y

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
p_gamma = fr1by2 * PC_gam_CrCr * (XCR - xe_gam_Cr)**2                      \
        +          PC_gam_CrNb * (XCR - xe_gam_Cr)    * (XNB - xe_gam_Nb)  \
        + fr1by2 * PC_gam_NbNb                        * (XNB - xe_gam_Nb)**2

p_delta = fr1by2 * PC_del_CrCr * (XCR - xe_del_Cr)**2                      \
        +          PC_del_CrNb * (XCR - xe_del_Cr)    * (XNB - xe_del_Nb)  \
        + fr1by2 * PC_del_NbNb                        * (XNB - xe_del_Nb)**2

p_laves = fr1by2 * PC_lav_CrCr * (XCR - xe_lav_Cr)**2                      \
        +          PC_lav_CrNb * (XCR - xe_lav_Cr)    * (XNB - xe_lav_Nb)  \
        + fr1by2 * PC_lav_NbNb                        * (XNB - xe_lav_Nb)**2

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


# ============ COMPOSITION SHIFTS ============
P_del, P_lav = symbols('P_del, P_lav')

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

A = Matrix([[GaCrCr       , GaCrNb       ,-GbCrCr       ,-GbCrNb       , 0            , 0            ],
            [GaCrNb       , GaNbNb       ,-GbCrNb       ,-GbNbNb       , 0            , 0            ],
            [GaCrCr       , GaCrNb       , 0            , 0            ,-GgCrCr       ,-GgCrNb       ],
            [GaCrNb       , GaNbNb       , 0            , 0            ,-GgCrNb       ,-GgNbNb       ],
            [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-DbCrCr-DbNbCr,-DbCrNb-DbNbNb, 0            , 0            ],
            [DaCrCr+DaNbCr, DaCrNb+DaNbNb, 0            , 0            ,-DgCrCr-DgNbCr,-DgCrNb-DgNbNb]])

A1 = Matrix([[ 0          , GaCrNb       ,-GbCrCr       ,-GbCrNb       , 0            , 0            ],
             [ 0          , GaNbNb       ,-GbCrNb       ,-GbNbNb       , 0            , 0            ],
             [ 0          , GaCrNb       , 0            , 0            ,-GgCrCr       ,-GgCrNb       ],
             [ 0          , GaNbNb       , 0            , 0            ,-GgCrNb       ,-GgNbNb       ],
             [-P_del      , DaCrNb+DaNbNb,-DbCrCr-DbNbCr,-DbCrNb-DbNbNb, 0            , 0            ],
             [-P_lav      , DaCrNb+DaNbNb, 0            , 0            ,-DgCrCr-DgNbCr,-DgCrNb-DgNbNb]])

A2 = Matrix([[GaCrCr       , 0            ,-GbCrCr       ,-GbCrNb       , 0            , 0            ],
             [GaCrNb       , 0            ,-GbCrNb       ,-GbNbNb       , 0            , 0            ],
             [GaCrCr       , 0            , 0            , 0            ,-GgCrCr       ,-GgCrNb       ],
             [GaCrNb       , 0            , 0            , 0            ,-GgCrNb       ,-GgNbNb       ],
             [DaCrCr+DaNbCr,-P_del        ,-DbCrCr-DbNbCr,-DbCrNb-DbNbNb, 0            , 0            ],
             [DaCrCr+DaNbCr,-P_lav        , 0            , 0            ,-DgCrCr-DgNbCr,-DgCrNb-DgNbNb]])

A3 = Matrix([[GaCrCr       , GaCrNb       , 0            ,-GbCrNb       , 0            , 0            ],
             [GaCrNb       , GaNbNb       , 0            ,-GbNbNb       , 0            , 0            ],
             [GaCrCr       , GaCrNb       , 0            , 0            ,-GgCrCr       ,-GgCrNb       ],
             [GaCrNb       , GaNbNb       , 0            , 0            ,-GgCrNb       ,-GgNbNb       ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-P_del        ,-DbCrNb-DbNbNb, 0            , 0            ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-P_lav        , 0            ,-DgCrCr-DgNbCr,-DgCrNb-DgNbNb]])

A4 = Matrix([[GaCrCr       , GaCrNb       ,-GbCrCr       , 0            , 0            , 0            ],
             [GaCrNb       , GaNbNb       ,-GbCrNb       , 0            , 0            , 0            ],
             [GaCrCr       , GaCrNb       , 0            , 0            ,-GgCrCr       ,-GgCrNb       ],
             [GaCrNb       , GaNbNb       , 0            , 0            ,-GgCrNb       ,-GgNbNb       ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-DbCrCr-DbNbCr,-P_del        , 0            , 0            ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb, 0            ,-P_lav        ,-DgCrCr-DgNbCr,-DgCrNb-DgNbNb]])

A5 = Matrix([[GaCrCr       , GaCrNb       ,-GbCrCr       ,-GbCrNb       , 0            , 0            ],
             [GaCrNb       , GaNbNb       ,-GbCrNb       ,-GbNbNb       , 0            , 0            ],
             [GaCrCr       , GaCrNb       , 0            , 0            , 0            ,-GgCrNb       ],
             [GaCrNb       , GaNbNb       , 0            , 0            , 0            ,-GgNbNb       ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-DbCrCr-DbNbCr,-DbCrNb-DbNbNb,-P_del        , 0            ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb, 0            , 0            ,-P_lav        ,-DgCrNb-DgNbNb]])

A6 = Matrix([[GaCrCr       , GaCrNb       ,-GbCrCr       ,-GbCrNb       , 0            , 0            ],
             [GaCrNb       , GaNbNb       ,-GbCrNb       ,-GbNbNb       , 0            , 0            ],
             [GaCrCr       , GaCrNb       , 0            , 0            ,-GgCrCr       , 0            ],
             [GaCrNb       , GaNbNb       , 0            , 0            ,-GgCrNb       , 0            ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb,-DbCrCr-DbNbCr,-DbCrNb-DbNbNb, 0            ,-P_del        ],
             [DaCrCr+DaNbCr, DaCrNb+DaNbNb, 0            , 0            ,-DgCrCr-DgNbCr,-P_lav        ]])

dA  = A.det()

dx_r_gam_Cr = A1.det() / dA
dx_r_gam_Nb = A2.det() / dA
dx_r_del_Cr = A3.det() / dA
dx_r_del_Nb = A4.det() / dA
dx_r_lav_Cr = A5.det() / dA
dx_r_lav_Nb = A6.det() / dA

# Generate numerically efficient C-code

codegen([# Equilibrium Compositions
         ('xe_gam_Cr', xe_gam_Cr),  ('xe_gam_Nb', xe_gam_Nb),
         ('xe_del_Cr', xe_del_Cr),  ('xe_del_Nb', xe_del_Nb),
         ('xe_lav_Cr', xe_lav_Cr),  ('xe_lav_Nb', xe_lav_Nb),
         # Curvature-Corrected Compositions
         ('xr_gam_Cr', xe_gam_Cr + dx_r_gam_Cr),  ('xr_gam_Nb', xe_gam_Nb + dx_r_gam_Nb),
         ('xr_del_Cr', xe_del_Cr + dx_r_del_Cr),  ('xr_del_Nb', xe_del_Nb + dx_r_del_Nb),
         ('xr_lav_Cr', xe_lav_Cr + dx_r_lav_Cr),  ('xr_lav_Nb', xe_lav_Nb + dx_r_lav_Nb),
         # Precipitate Properties
         ('r_delta', r_delta),  ('r_laves', r_laves),
         ('s_delta', s_delta),  ('s_laves', s_laves),
         # Gibbs energies
         ('g_gam', p_gamma),  ('g_del', p_delta),  ('g_lav', p_laves),
         # First derivatives
         ('dg_gam_dxCr', p_dGgam_dxCr),  ('dg_gam_dxNb', p_dGgam_dxNb),
         ('dg_del_dxCr', p_dGdel_dxCr),  ('dg_del_dxNb', p_dGdel_dxNb),
         ('dg_lav_dxCr', p_dGlav_dxCr),  ('dg_lav_dxNb', p_dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr', p_d2Ggam_dxCrCr), ('d2g_gam_dxCrNb', p_d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr', p_d2Ggam_dxNbCr), ('d2g_gam_dxNbNb', p_d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr', p_d2Gdel_dxCrCr), ('d2g_del_dxCrNb', p_d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr', p_d2Gdel_dxNbCr), ('d2g_del_dxNbNb', p_d2Gdel_dxNbNb),
         ('d2g_lav_dxCrCr', p_d2Glav_dxCrCr), ('d2g_lav_dxCrNb', p_d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr', p_d2Glav_dxNbCr), ('d2g_lav_dxNbNb', p_d2Glav_dxNbNb)],
        language='C', prefix='parabola625', project='PrecipitateAging', to_files=True)

# Generate numerically efficient Python code
leverCr = lambdify([xo, yo, xb, yb, xc, yc, xd, yd], levers[y])
leverNb = lambdify([xo, yo, xb, yb, xc, yc, xd, yd], levers[x])

PG = lambdify([XCR, XNB], p_gamma)
PD = lambdify([XCR, XNB], p_delta)
PL = lambdify([XCR, XNB], p_laves)

DXAB = lambdify([P_del, P_lav], dx_r_gam_Cr) # Cr in gamma phase
DXAC = lambdify([P_del, P_lav], dx_r_gam_Nb) # Nb in gamma phase
DXBB = lambdify([P_del, P_lav], dx_r_del_Cr) # Cr in delta phase
DXBC = lambdify([P_del, P_lav], dx_r_del_Nb) # Nb in delta phase
DXGB = lambdify([P_del, P_lav], dx_r_lav_Cr) # Cr in Laves phase
DXGC = lambdify([P_del, P_lav], dx_r_lav_Nb) # Nb in Laves phase
