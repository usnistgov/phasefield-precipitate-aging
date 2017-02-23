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
# - μ as $\mathrm{Nb_6(Cr, Nb, Ni)_7}$
# - Laves as $\mathrm{(\mathbf{Cr}, Nb, Ni)_2(Cr, \mathbf{Nb})_1}$
# 
# The phase field model requires Gibbs free energies as functions of system
# compositions $x_\mathrm{Cr}$, $x_\mathrm{Nb}$, $x_\mathrm{Ni}$. The Calphad
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
#      * $y_\mathrm{Ni}'  = 1-4x_\mathrm{Nb}$
#      * $y_\mathrm{Cr}'' = \frac{4}{3}x_\mathrm{Cr}$
#      * $y_\mathrm{Ni}'' = 1-\frac{4}{3}x_\mathrm{Cr}$
#      * Constraints: $x_\mathrm{Nb}\leq\frac{1}{4}$
#                     $x_\mathrm{Cr}\leq\frac{3}{4}$
#
# - μ: no changes necessary
#      * $y_\mathrm{Nb}'  = 1$
#      * $y_\mathrm{Cr}'' = x_\mathrm{Cr}$
#      * $y_\mathrm{Nb}'' = \frac{13}{7}x_\mathrm{Nb}-\frac{6}{7}$
#      * $y_\mathrm{Ni}'' = x_\mathrm{Ni}$
#      * Constraints: $x_\mathrm{Cr}\leq\frac{7}{13}$ 
#                     $x_\mathrm{Nb}\geq\frac{6}{13}$
#                     $x_\mathrm{Ni}\leq\frac{7}{13}$
#
# - Laves: eliminate Nb from the first (Cr) sublattice, $\mathrm{(\mathbf{Cr}, Ni)_2(Cr, \mathbf{Nb})_1}$
#      * $y_\mathrm{Cr}'  = 1-\frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Ni}'  = \frac{3}{2}x_\mathrm{Ni}$
#      * $y_\mathrm{Cr}'' = 1-3x_\mathrm{Nb}$
#      * $y_\mathrm{Nb}'' = 3x_\mathrm{Nb}$
#      * Constraints: $0\leq x_\mathrm{Ni}\leq\frac{2}{3}$
#                     $0\leq x_\mathrm{Nb}\leq\frac{1}{3}$

# Numerical libraries
import numpy as np
from scipy.optimize import fsolve
from sympy.utilities.lambdify import lambdify
from sympy.printing.theanocode import theano_function
from scipy.spatial import ConvexHull

# Runtime / parallel libraries
import time
import warnings

# Thermodynamics and computer-algebra libraries
from pycalphad import Database, calculate, Model
from sympy.utilities.codegen import codegen
from sympy.parsing.sympy_parser import parse_expr
from sympy import And, Ge, Gt, Le, Lt, Or, Piecewise, true
from sympy import Abs, cse, diff, logcombine, powsimp, simplify, symbols, sympify
from sympy import exp

# Visualization libraries
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm

# Constants
epsilon = 1e-10 # tolerance for comparing floating-point numbers to zero
temp = 870.0 + 273.15 # 1143 Kelvin

RT = 8.3144598*temp # J/mol/K
Vm = 1.0e-5 # m^3/mol
inVm = 1.0 / Vm # mol/m^3

# Let's avoid integer arithmetic in fractions.
fr13by7 = 13.0/7
fr13by3 = 13.0/3
fr13by4 = 13.0/4
fr6by7 = 6.0/7
fr6by13 = 6.0/13
fr7by13 = 7.0/13
fr3by4 = 3.0/4
fr3by2 = 3.0/2
fr4by3 = 4.0/3
fr2by3 = 2.0/3
fr1by3 = 1.0/3
fr1by2 = 1.0/2
rt3by2 = np.sqrt(3.0)/2

# Helper functions to convert compositions into (x,y) coordinates
def simX(x2, x3):
    return x2 + fr1by2 * x3

def simY(x3):
    return rt3by2 * x3

# triangle bounding the Gibbs simplex
XS = [0.0, simX(1,0), simX(0,1), 0.0]
YS = [0.0, simY(0),   simY(1),   0.0]

# triangle bounding three-phase coexistence
XT = [0.25, simX(0.4875,0.025), simX(0.5375,0.4625), 0.25]
YT = [0.0,  simY(0.025),        simY(0.4625),        0.0]

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

species = list(set([i for c in tdb.phases['D85_NI7NB6'].constituents for i in c]))
model = Model(tdb, species, 'D85_NI7NB6')
g_mu = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases['C14_LAVES'].constituents for i in c]))
model = Model(tdb, species, 'C14_LAVES')
g_laves = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases['C15_LAVES'].constituents for i in c]))
model = Model(tdb, species, 'C15_LAVES')
g_lavesLT = parse_expr(str(model.ast))

species = list(set([i for c in tdb.phases['BCC_A2'].constituents for i in c]))
model = Model(tdb, species, 'BCC_A2')
g_bcc = parse_expr(str(model.ast))


# Convert sublattice to phase composition (y to x)
# Declare sublattice variables used in Pycalphad expressions
# Gamma
FCC_A10CR, FCC_A10NB, FCC_A10NI, FCC_A11VA = symbols('FCC_A10CR FCC_A10NB FCC_A10NI FCC_A11VA')
# Delta
D0A_NBNI30NI, D0A_NBNI30NB, D0A_NBNI31CR, D0A_NBNI31NI = symbols('D0A_NBNI30NI D0A_NBNI30NB D0A_NBNI31CR D0A_NBNI31NI')
# Mu
D85_NI7NB60NB, D85_NI7NB61CR, D85_NI7NB61NB, D85_NI7NB61NI = symbols('D85_NI7NB60NB D85_NI7NB61CR D85_NI7NB61NB D85_NI7NB61NI')
# Laves
C14_LAVES0CR, C14_LAVES0NI, C14_LAVES1CR, C14_LAVES1NB = symbols('C14_LAVES0CR C14_LAVES0NI C14_LAVES1CR C14_LAVES1NB') 
C15_LAVES0CR, C15_LAVES0NI, C15_LAVES1CR, C15_LAVES1NB = symbols('C15_LAVES0CR C15_LAVES0NI C15_LAVES1CR C15_LAVES1NB') 
# Temperature
T = symbols('T')

# Declare system variables for target expressions
GAMMA_XCR, GAMMA_XNB, GAMMA_XNI = symbols('GAMMA_XCR GAMMA_XNB GAMMA_XNI')
DELTA_XCR, DELTA_XNB, DELTA_XNI = symbols('DELTA_XCR DELTA_XNB DELTA_XNI')
MU_XCR, MU_XNB, MU_XNI = symbols('MU_XCR MU_XNB MU_XNI')
LAVES_XCR, LAVES_XNB, LAVES_XNI = symbols('LAVES_XCR LAVES_XNB LAVES_XNI')
BCC_XCR, BCC_XNB, BCC_XNI = symbols('BCC_XCR BCC_XNB BCC_XNI')

# Specify equilibrium points for phase field
xe_gam_Cr= 0.3333
xe_gam_Nb= 0.0333
xe_gam_Ni= 1.0 - xe_gam_Cr - xe_gam_Nb

xe_del_Cr = 0.0125
xe_del_Nb = 0.2475

xe_mu_Cr  = 0.0250
xe_mu_Nb  = 0.5250
xe_mu_Ni  = 1.0 - xe_mu_Cr - xe_mu_Nb

xe_lav_Cr = 0.3500
xe_lav_Nb = 0.2750
xe_lav_Ni = 1.0 - xe_lav_Cr - xe_lav_Nb

# Anchor points for Taylor series
X0 = [simX(xe_gam_Nb, xe_gam_Cr), simX(xe_del_Nb, xe_del_Cr), simX(xe_mu_Nb, xe_mu_Cr), simX(xe_lav_Nb, xe_lav_Cr)]
Y0 = [simY(xe_gam_Cr), simY(xe_del_Cr), simY(xe_mu_Cr), simY(xe_lav_Cr)]


# Make sublattice -> system substitutions
g_gamma = inVm * g_gamma.subs({FCC_A10CR: GAMMA_XCR,
                               FCC_A10NB: GAMMA_XNB,
                               FCC_A10NI: GAMMA_XNI,
                               FCC_A11VA: 1.0,
                               T: temp})

g_delta = inVm * g_delta.subs({D0A_NBNI30NB: 4.0*DELTA_XNB,
                               D0A_NBNI30NI: 1.0 - 4.0*DELTA_XNB,
                               D0A_NBNI31CR: fr4by3 * DELTA_XCR,
                               D0A_NBNI31NI: 1.0 - fr4by3 * DELTA_XCR,
                               T: temp})

g_mu = inVm * g_mu.subs({D85_NI7NB60NB: 1,
                         D85_NI7NB61CR: fr13by7*MU_XCR,
                         D85_NI7NB61NB: fr13by7*MU_XNB - fr6by7,
                         D85_NI7NB61NI: fr13by7*MU_XNI,
                         T: temp})

g_laves = inVm * g_laves.subs({C14_LAVES0CR: 1.0 - fr3by2*LAVES_XNI,
                               C14_LAVES0NI: fr3by2 * LAVES_XNI,
                               C14_LAVES1CR: 1.0 - 3.0*LAVES_XNB,
                               C14_LAVES1NB: 3.0 * LAVES_XNB,
                               T: temp})

g_lavesLT = inVm * g_lavesLT.subs({C15_LAVES0CR: 1.0 - fr3by2*LAVES_XNI,
                                C15_LAVES0NI: fr3by2 * LAVES_XNI,
                                C15_LAVES1CR: 1.0 - 3.0*LAVES_XNB,
                                C15_LAVES1NB: 3.0 * LAVES_XNB,
                                T: temp})

# Create parabolic approximation functions
p_gamma = g_gamma.subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
        + 0.0625 * diff(g_gamma, GAMMA_XCR, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XCR - xe_gam_Cr)**2\
        + 1.5000 * diff(g_gamma, GAMMA_XNB, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XNB - xe_gam_Nb)**2

p_delta = g_delta.subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) \
        + 0.2000 * diff(g_delta, DELTA_XCR, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XCR - xe_del_Cr)**2 \
        + 6.0000 * diff(g_delta, DELTA_XNB, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XNB - xe_del_Nb)**2

p_mu    = g_mu.subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
        + 0.1750 * diff(g_mu, MU_XCR, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XCR    - xe_mu_Cr )**2 \
        + 1.0000 * diff(g_mu, MU_XNB, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XNB    - xe_mu_Nb )**2

p_laves = g_laves.subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
        + 0.5000 * diff(g_laves, LAVES_XNB, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNB - xe_lav_Nb)**2 \
        + 0.5000 * diff(g_laves, LAVES_XNI, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNI - xe_lav_Ni)**2

# Generate first derivatives
p_dGgam_dxCr = diff(p_gamma, GAMMA_XCR)
p_dGgam_dxNb = diff(p_gamma, GAMMA_XNB)
p_dGgam_dxNi = diff(p_gamma, GAMMA_XNI)

p_dGdel_dxCr = diff(p_delta, DELTA_XCR)
p_dGdel_dxNb = diff(p_delta, DELTA_XNB)

p_dGmu_dxCr = diff(p_mu, MU_XCR)
p_dGmu_dxNb = diff(p_mu, MU_XNB)

p_dGlav_dxCr = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
p_dGlav_dxNb = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)

# Generate second derivatives
p_d2Ggam_dxCrCr = diff(p_gamma, GAMMA_XCR, GAMMA_XCR)
p_d2Ggam_dxCrNb = diff(p_gamma, GAMMA_XCR, GAMMA_XNB)
p_d2Ggam_dxNbCr = diff(p_gamma, GAMMA_XNB, GAMMA_XCR)
p_d2Ggam_dxNbNb = diff(p_gamma, GAMMA_XNB, GAMMA_XNB)

p_d2Gdel_dxCrCr = diff(p_delta, DELTA_XCR, DELTA_XCR)
p_d2Gdel_dxCrNb = diff(p_delta, DELTA_XCR, DELTA_XNB)
p_d2Gdel_dxNbCr = diff(p_delta, DELTA_XNB, DELTA_XCR)
p_d2Gdel_dxNbNb = diff(p_delta, DELTA_XNB, DELTA_XNB)

p_d2Gmu_dxCrCr = diff(p_mu, MU_XCR, MU_XCR)
p_d2Gmu_dxCrNb = diff(p_mu, MU_XCR, MU_XNB)
p_d2Gmu_dxNbCr = diff(p_mu, MU_XNB, MU_XCR)
p_d2Gmu_dxNbNb = diff(p_mu, MU_XNB, MU_XNB)

p_d2Glav_dxCrCr = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XCR)
p_d2Glav_dxCrNb = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XNB)
p_d2Glav_dxNbCr = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XCR)
p_d2Glav_dxNbNb = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XNB)

# Write paraboloid energies as C functions
codegen([# Gibbs energies
         ('g_gam', p_gamma),
         ('g_del', p_delta),
         ('g_mu',  p_mu),
         ('g_lav', p_laves),
         # Constants
         ('xe_gam_Cr', xe_gam_Cr), ('xe_gam_Nb', xe_gam_Nb),
         ('xe_del_Cr', xe_del_Cr), ('xe_del_Nb', xe_del_Nb),
         ('xe_mu_Cr',  xe_mu_Cr),  ('xe_mu_Nb',  xe_mu_Nb),
         ('xe_lav_Nb', xe_lav_Nb), ('xe_lav_Ni', xe_lav_Ni),
         # First derivatives
         ('dg_gam_dxCr', p_dGgam_dxCr), ('dg_gam_dxNb', p_dGgam_dxNb),
         ('dg_gam_dxNi', p_dGgam_dxNi),
         ('dg_del_dxCr', p_dGdel_dxCr), ('dg_del_dxNb', p_dGdel_dxNb),
         ('dg_mu_dxCr',  p_dGmu_dxCr),  ('dg_mu_dxNb',  p_dGmu_dxNb),
         ('dg_lav_dxCr', p_dGlav_dxCr), ('dg_lav_dxNb', p_dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr', p_d2Ggam_dxCrCr), ('d2g_gam_dxCrNb', p_d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr', p_d2Ggam_dxNbCr), ('d2g_gam_dxNbNb', p_d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr', p_d2Gdel_dxCrCr), ('d2g_del_dxCrNb', p_d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr', p_d2Gdel_dxNbCr), ('d2g_del_dxNbNb', p_d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',  p_d2Gmu_dxCrCr),  ('d2g_mu_dxCrNb',  p_d2Gmu_dxCrNb),
         ('d2g_mu_dxNbCr',  p_d2Gmu_dxNbCr),  ('d2g_mu_dxNbNb',  p_d2Gmu_dxNbNb),
         ('d2g_lav_dxCrCr', p_d2Glav_dxCrCr), ('d2g_lav_dxCrNb', p_d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr', p_d2Glav_dxNbCr), ('d2g_lav_dxNbNb', p_d2Glav_dxNbNb)],
         language='C', prefix='parabola625', project='ALLOY625', to_files=True)

# Create Taylor series expansions
t_gamma = g_gamma.subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
        + 1.0 * diff(g_gamma, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XCR - xe_gam_Cr) \
        + 1.0 * diff(g_gamma, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XNB - xe_gam_Nb) \
        + 1.0 * diff(g_gamma, GAMMA_XNI).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XNI - xe_gam_Ni) \
        + 0.5 * diff(g_gamma, GAMMA_XCR, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XCR - xe_gam_Cr)**2 \
        + 0.5 * diff(g_gamma, GAMMA_XNB, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XNB - xe_gam_Nb)**2 \
        + 0.5 * diff(g_gamma, GAMMA_XNI, GAMMA_XNI).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) * (GAMMA_XNI - xe_gam_Ni)**2 \
        + 0.5 * ( diff(g_gamma, GAMMA_XCR, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                + diff(g_gamma, GAMMA_XNB, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                ) * (GAMMA_XCR - xe_gam_Cr) * (GAMMA_XNB - xe_gam_Nb) \
        + 0.5 * ( diff(g_gamma, GAMMA_XCR, GAMMA_XNI).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                + diff(g_gamma, GAMMA_XNI, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                ) * (GAMMA_XCR - xe_gam_Cr) * (GAMMA_XNI - xe_gam_Ni) \
        + 0.5 * ( diff(g_gamma, GAMMA_XNB, GAMMA_XNI).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                + diff(g_gamma, GAMMA_XNI, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni}) \
                ) * (GAMMA_XNB - xe_gam_Nb) * (GAMMA_XNI - xe_gam_Ni)

t_delta = g_delta.subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) \
        + 1.0 * diff(g_delta, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XCR - xe_del_Cr) \
        + 1.0 * diff(g_delta, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XNB - xe_del_Nb) \
        + 0.5 * diff(g_delta, DELTA_XCR, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XCR - xe_del_Cr)**2 \
        + 0.5 * diff(g_delta, DELTA_XNB, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) * (DELTA_XNB - xe_del_Nb)**2 \
        + 0.5 * ( diff(g_delta, DELTA_XCR, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) \
                + diff(g_delta, DELTA_XNB, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb}) \
                ) * (DELTA_XCR - xe_del_Cr) * (DELTA_XNB - xe_del_Nb)

t_mu    = g_mu.subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
        + 1.0 * diff(g_mu, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XCR    - xe_mu_Cr ) \
        + 1.0 * diff(g_mu, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XNB    - xe_mu_Nb ) \
        + 1.0 * diff(g_mu, MU_XNI).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XNI    - xe_mu_Ni ) \
        + 0.5 * diff(g_mu, MU_XCR, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XCR    - xe_mu_Cr )**2 \
        + 0.5 * diff(g_mu, MU_XNB, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XNB    - xe_mu_Nb )**2 \
        + 0.5 * diff(g_mu, MU_XNI, MU_XNI).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) * (MU_XNI    - xe_mu_Ni )**2 \
        + 0.5 * ( diff(g_mu, MU_XCR, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                + diff(g_mu, MU_XNB, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                ) * (MU_XCR    - xe_mu_Cr ) * (MU_XNB    - xe_mu_Nb ) \
        + 0.5 * ( diff(g_mu, MU_XCR, MU_XNI).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                + diff(g_mu, MU_XNI, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                ) * (MU_XCR    - xe_mu_Cr ) * (MU_XNI    - xe_mu_Ni ) \
        + 0.5 * ( diff(g_mu, MU_XNB, MU_XNI).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                + diff(g_mu, MU_XNI, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni}) \
                ) * (MU_XNB    - xe_mu_Nb ) * (MU_XNI    - xe_mu_Ni )

t_laves = g_laves.subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
        + 1.0 * diff(g_laves, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNB - xe_lav_Nb) \
        + 1.0 * diff(g_laves, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNI - xe_lav_Ni) \
        + 0.5 * diff(g_laves, LAVES_XNB, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNB - xe_lav_Nb)**2 \
        + 0.5 * diff(g_laves, LAVES_XNI, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNI - xe_lav_Ni)**2 \
        + 0.5 * ( diff(g_laves, LAVES_XNB, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
                + diff(g_laves, LAVES_XNI, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
                ) * (LAVES_XNB - xe_lav_Nb) * (LAVES_XNI - xe_lav_Ni)

t_lavesLT = g_lavesLT.subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
        + 1.0 * diff(g_lavesLT, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNB - xe_lav_Nb) \
        + 1.0 * diff(g_lavesLT, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNI - xe_lav_Ni) \
        + 0.5 * diff(g_lavesLT, LAVES_XNB, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNB - xe_lav_Nb)**2 \
        + 0.5 * diff(g_lavesLT, LAVES_XNI, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) * (LAVES_XNI - xe_lav_Ni)**2 \
        + 0.5 * ( diff(g_lavesLT, LAVES_XNB, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
                + diff(g_lavesLT, LAVES_XNI, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni}) \
                ) * (LAVES_XNB - xe_lav_Nb) * (LAVES_XNI - xe_lav_Ni)

# Generate first derivatives
t_dGgam_dxCr = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR)
t_dGgam_dxNb = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB)
t_dGgam_dxNi = diff(t_gamma, GAMMA_XNI)

t_dGdel_dxCr = diff(t_delta, DELTA_XCR)
t_dGdel_dxNb = diff(t_delta, DELTA_XNB)

t_dGmu_dxCr = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR)
t_dGmu_dxNb = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB)

t_dGlav_dxCr = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
t_dGlav_dxNb = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)

# Generate second derivatives
t_d2Ggam_dxCrCr = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR, GAMMA_XCR)
t_d2Ggam_dxCrNb = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR, GAMMA_XNB)
t_d2Ggam_dxNbCr = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB, GAMMA_XCR)
t_d2Ggam_dxNbNb = diff(t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB, GAMMA_XNB)

t_d2Gdel_dxCrCr = diff(t_delta, DELTA_XCR, DELTA_XCR)
t_d2Gdel_dxCrNb = diff(t_delta, DELTA_XCR, DELTA_XNB)
t_d2Gdel_dxNbCr = diff(t_delta, DELTA_XNB, DELTA_XCR)
t_d2Gdel_dxNbNb = diff(t_delta, DELTA_XNB, DELTA_XNB)

t_d2Gmu_dxCrCr = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR, MU_XCR)
t_d2Gmu_dxCrNb = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR, MU_XNB)
t_d2Gmu_dxNbCr = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB, MU_XCR)
t_d2Gmu_dxNbNb = diff(t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB, MU_XNB)

t_d2Glav_dxCrCr = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XCR)
t_d2Glav_dxCrNb = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XNB)
t_d2Glav_dxNbCr = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XCR)
t_d2Glav_dxNbNb = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XNB)

# Write Taylor series functions as C code
codegen([# Gibbs energies
         ('g_gam', t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('g_del', t_delta),
         ('g_mu',  t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB})),
         ('g_lav', t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB})),
         # Constants
         ('xe_gam_Cr', xe_gam_Cr), ('xe_gam_Nb', xe_gam_Nb),
         ('xe_del_Cr', xe_del_Cr), ('xe_del_Nb', xe_del_Nb),
         ('xe_mu_Cr',  xe_mu_Cr),  ('xe_mu_Nb',  xe_mu_Nb),
         ('xe_lav_Nb', xe_lav_Nb), ('xe_lav_Ni', xe_lav_Ni),
         # First derivatives
         ('dg_gam_dxCr', t_dGgam_dxCr), ('dg_gam_dxNb', t_dGgam_dxNb),
         ('dg_gam_dxNi', t_dGgam_dxNi.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('dg_del_dxCr',    t_dGdel_dxCr), ('dg_del_dxNb', t_dGdel_dxNb),
         ('dg_mu_dxCr',     t_dGmu_dxCr),  ('dg_mu_dxNb',  t_dGmu_dxNb),
         ('dg_lav_dxCr',    t_dGlav_dxCr), ('dg_lav_dxNb', t_dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr', t_d2Ggam_dxCrCr), ('d2g_gam_dxCrNb', t_d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr', t_d2Ggam_dxNbCr), ('d2g_gam_dxNbNb', t_d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr', t_d2Gdel_dxCrCr), ('d2g_del_dxCrNb', t_d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr', t_d2Gdel_dxNbCr), ('d2g_del_dxNbNb', t_d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',  t_d2Gmu_dxCrCr),  ('d2g_mu_dxCrNb',  t_d2Gmu_dxCrNb),
         ('d2g_mu_dxNbCr',  t_d2Gmu_dxNbCr),  ('d2g_mu_dxNbNb',  t_d2Gmu_dxNbNb),
         ('d2g_lav_dxCrCr', t_d2Glav_dxCrCr), ('d2g_lav_dxCrNb', t_d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr', t_d2Glav_dxNbCr), ('d2g_lav_dxNbNb', t_d2Glav_dxNbNb)],
        language='C', prefix='taylor625', project='ALLOY625', to_files=True)

# Generate safe CALPHAD expressions
c_gamma = Piecewise((g_gamma,
                     Gt(GAMMA_XCR, 0) & Lt(GAMMA_XCR, 1) &
                     Gt(GAMMA_XNB, 0) & Lt(GAMMA_XNB, 1) &
                     Gt(GAMMA_XNI, 0) & Lt(GAMMA_XNI, 1)),
                    (t_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), True))

c_delta = Piecewise((g_delta,
                     Gt(DELTA_XCR, 0)             & Le(DELTA_XCR, 0.75)          &
                     Gt(DELTA_XNB, 0)             & Le(DELTA_XNB, 0.25)          &
                     Gt(1.0-DELTA_XCR-DELTA_XNB, 0) & Lt(1.0-DELTA_XCR-DELTA_XNB, 1)),
                    (t_delta, True))

c_mu    = Piecewise((g_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}),
                     Gt(MU_XCR, 0)          & Le(MU_XCR, fr7by13)          &
                     Ge(MU_XNB, fr6by13)    & Lt(MU_XNB, 1)                &
                     Gt(1.0-MU_XCR-MU_XNB, 0) & Le(1.0-MU_XCR-MU_XNB, fr7by13)),
                    (t_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), True))

c_laves = Piecewise((g_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}),
                     Gt(LAVES_XCR, 0) & Lt(LAVES_XCR, 1)             &
                     Gt(LAVES_XNB, 0) & Le(LAVES_XNB, fr1by3)        &
                     Gt(LAVES_XNI, 0) & Le(LAVES_XNI, fr2by3))       ,
                    (t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), True))

c_lavesLT = Piecewise((g_lavesLT.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}),
                     Gt(LAVES_XCR, 0) & Lt(LAVES_XCR, 1)             &
                     Gt(LAVES_XNB, 0) & Le(LAVES_XNB, fr1by3)        &
                     Gt(LAVES_XNI, 0) & Le(LAVES_XNI, fr2by3))       ,
                    (t_lavesLT.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), True))

# Generate first derivatives
dGgam_dxCr = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR)
dGgam_dxNb = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB)
dGgam_dxNi = diff(c_gamma, GAMMA_XNI)

dGdel_dxCr = diff(c_delta, DELTA_XCR)
dGdel_dxNb = diff(c_delta, DELTA_XNB)

dGmu_dxCr = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR)
dGmu_dxNb = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB)

dGlav_dxCr = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
dGlav_dxNb = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)

# Generate optimized second derivatives
d2Ggam_dxCrCr = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR, GAMMA_XCR)
d2Ggam_dxCrNb = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XCR, GAMMA_XNB)
d2Ggam_dxNbCr = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB, GAMMA_XCR)
d2Ggam_dxNbNb = diff(c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB}), GAMMA_XNB, GAMMA_XNB)

d2Gdel_dxCrCr = diff(c_delta, DELTA_XCR, DELTA_XCR)
d2Gdel_dxCrNb = diff(c_delta, DELTA_XCR, DELTA_XNB)
d2Gdel_dxNbCr = diff(c_delta, DELTA_XNB, DELTA_XCR)
d2Gdel_dxNbNb = diff(c_delta, DELTA_XNB, DELTA_XNB)

d2Gmu_dxCrCr = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR, MU_XCR)
d2Gmu_dxCrNb = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XCR, MU_XNB)
d2Gmu_dxNbCr = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB, MU_XCR)
d2Gmu_dxNbNb = diff(c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB}), MU_XNB, MU_XNB)

d2Glav_dxCrCr = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XCR)
d2Glav_dxCrNb = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR, LAVES_XNB)
d2Glav_dxNbCr = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XCR)
d2Glav_dxNbNb = diff(c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB, LAVES_XNB)

# Write CALPHAD functions as C code
codegen([# Gibbs energies
         ('g_gam', c_gamma.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('g_del', c_delta),
         ('g_mu',  c_mu.subs({MU_XNI: 1.0-MU_XCR-MU_XNB})),
         ('g_lav', c_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB})),
         # Constants
         ('xe_gam_Cr', xe_gam_Cr), ('xe_gam_Nb', xe_gam_Nb),
         ('xe_del_Cr', xe_del_Cr), ('xe_del_Nb', xe_del_Nb),
         ('xe_mu_Cr',  xe_mu_Cr),  ('xe_mu_Nb',  xe_mu_Nb),
         ('xe_lav_Nb', xe_lav_Nb), ('xe_lav_Ni', xe_lav_Ni),
         # First derivatives
         ('dg_gam_dxCr', dGgam_dxCr.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('dg_gam_dxNb', dGgam_dxNb.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('dg_gam_dxNi', dGgam_dxNi.subs({GAMMA_XNI: 1.0-GAMMA_XCR-GAMMA_XNB})),
         ('dg_del_dxCr', dGdel_dxCr), ('dg_del_dxNb', dGdel_dxNb),
         ('dg_mu_dxCr',  dGmu_dxCr),  ('dg_mu_dxNb',  dGmu_dxNb),
         ('dg_lav_dxCr', dGlav_dxCr), ('dg_lav_dxNb', dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr', d2Ggam_dxCrCr), ('d2g_gam_dxCrNb', d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr', d2Ggam_dxNbCr), ('d2g_gam_dxNbNb', d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr', d2Gdel_dxCrCr), ('d2g_del_dxCrNb', d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr', d2Gdel_dxNbCr), ('d2g_del_dxNbNb', d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',  d2Gmu_dxCrCr),  ('d2g_mu_dxCrNb',  d2Gmu_dxCrNb),
         ('d2g_mu_dxNbCr',  d2Gmu_dxNbCr),  ('d2g_mu_dxNbNb',  d2Gmu_dxNbNb),
         ('d2g_lav_dxCrCr', d2Glav_dxCrCr), ('d2g_lav_dxCrNb', d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr', d2Glav_dxNbCr), ('d2g_lav_dxNbNb', d2Glav_dxNbNb)],
        language='C', prefix='energy625', project='ALLOY625', to_files=True)

print "Finished writing paraboloid, Taylor series, and CALPHAD energy functions to disk."

# Generate numerically efficient system-composition expressions
TG = lambdify([GAMMA_XCR, GAMMA_XNB], t_gamma.subs({GAMMA_XNI: 1-GAMMA_XCR-GAMMA_XNB}), modules='sympy')
TD = lambdify([DELTA_XCR, DELTA_XNB], t_delta, modules='sympy')
TU = lambdify([MU_XCR,    MU_XNB],    t_mu.subs({MU_XNI: 1-MU_XCR-MU_XNB}), modules='sympy')
TL = lambdify([LAVES_XCR, LAVES_XNB], t_laves.subs({LAVES_XNI: 1-LAVES_XCR-LAVES_XNB}), modules='sympy')
TLL= lambdify([LAVES_XCR, LAVES_XNB], t_lavesLT.subs({LAVES_XNI: 1-LAVES_XCR-LAVES_XNB}), modules='sympy')

GG = lambdify([GAMMA_XCR, GAMMA_XNB], c_gamma.subs({GAMMA_XNI: 1-GAMMA_XCR-GAMMA_XNB}), modules='sympy')
GD = lambdify([DELTA_XCR, DELTA_XNB], c_delta, modules='sympy')
GU = lambdify([MU_XCR,    MU_XNB],    c_mu.subs({MU_XNI: 1-MU_XCR-MU_XNB}), modules='sympy')
GL = lambdify([LAVES_XCR, LAVES_XNB], c_laves.subs({LAVES_XNI: 1-LAVES_XCR-LAVES_XNB}), modules='sympy')
GLL= lambdify([LAVES_XCR, LAVES_XNB], c_lavesLT.subs({LAVES_XNI: 1-LAVES_XCR-LAVES_XNB}), modules='sympy')

print "Finished lambdifying Taylor series and CALPHAD energy functions."
