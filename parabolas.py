#!/usr/bin/python

# Numerical libraries
import numpy as np
from scipy.optimize import fsolve
from sympy.utilities.lambdify import lambdify
from scipy.spatial import ConvexHull

# Runtime / parallel libraries
import time
import warnings
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool

# Thermodynamics and computer-algebra libraries
from pycalphad import Database, calculate, Model
from sympy.utilities.codegen import codegen
from sympy.parsing.sympy_parser import parse_expr
from sympy import And, Ge, Gt, Le, Lt, Or, Piecewise, true
from sympy import Abs, cse, diff, logcombine, powsimp, simplify, symbols, sympify

# Visualization libraries
import matplotlib.pylab as plt
#from ipywidgets import FloatProgress
#from IPython.display import display

# Constants
epsilon = 1e-10 # tolerance for comparing floating-point numbers to zero
dx = 0.1 # small offset to avoid valid roots at edges of defined regions
temp = 870.0 + 273.15 # 1143 Kelvin
#temp = 1150.0 + 273.15 # Kelvin

RT = 8.3144598*temp # J/mol/K
Vm = 1.0e-5 # m^3/mol
inVm = 1.0 / Vm # mol/m^3
errslope = 1.5e5 # J/(mol/mol)m curvature of parabola outside phase-specific domain

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

# Read CALPHAD database from disk, specify phases and elements of interest
tdb = Database('Du_Cr-Nb-Ni_simple.tdb')
phases = ['FCC_A1', 'D0A_NBNI3', 'D85_NI7NB6', 'C14_LAVES', 'C15_LAVES', 'BCC_A2']
elements = ['CR', 'NB', 'NI']

c_gamma = list(set([i for c in tdb.phases['FCC_A1'].constituents for i in c]))
m_gamma = Model(tdb, c_gamma, 'FCC_A1')
g_gamma = parse_expr(str(m_gamma.ast))

c_delta = list(set([i for c in tdb.phases['D0A_NBNI3'].constituents for i in c]))
m_delta = Model(tdb, c_delta, 'D0A_NBNI3')
g_delta = parse_expr(str(m_delta.ast))

c_mu = list(set([i for c in tdb.phases['D85_NI7NB6'].constituents for i in c]))
m_mu = Model(tdb, c_mu, 'D85_NI7NB6')
g_mu = parse_expr(str(m_mu.ast))

c_lavesHT = list(set([i for c in tdb.phases['C14_LAVES'].constituents for i in c]))
m_lavesHT = Model(tdb, c_lavesHT, 'C14_LAVES')
g_laves = parse_expr(str(m_lavesHT.ast))

c_lavesLT = list(set([i for c in tdb.phases['C15_LAVES'].constituents for i in c]))
m_lavesLT = Model(tdb, c_lavesLT, 'C15_LAVES')
g_lavesLT = parse_expr(str(m_lavesLT.ast))

c_bcc = list(set([i for c in tdb.phases['BCC_A2'].constituents for i in c]))
m_bcc = Model(tdb, c_bcc, 'BCC_A2')
g_bcc = parse_expr(str(m_bcc.ast))


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
#xe_gam_Cr = 0.475 # original
#xe_gam_Nb = 0.02
#xe_gam_Cr = 0.34588 # multimin
#xe_gam_Nb = 0.18221
xe_gam_Cr = 1.0e-2 # landscape
xe_gam_Nb = 1.0/3 - 1.0e-2
xe_gam_Ni = 1.0 - xe_gam_Cr - xe_gam_Nb

#xe_del_Cr = 0.0125 # original
#xe_del_Nb = 0.2375
#xe_del_Cr = 0.01015 # multimin
#xe_del_Nb = 0.25000
xe_del_Cr = 5.0e-3 # landscape
xe_del_Nb = 0.25 - 5.0e-3

#xe_mu_Cr = 0.025 # original
#xe_mu_Nb = 0.4875
#xe_mu_Cr = 0.02792 # multimin
#xe_mu_Nb = 0.49076
xe_mu_Cr = 7.5e-3 # landscape
xe_mu_Nb = 0.5 + 7.5e-3
xe_mu_Ni = 1.0 - xe_mu_Cr - xe_mu_Nb

#xe_lav_Nb = 0.2625 # original
#xe_lav_Ni = 0.375
#xe_lav_Nb = 0.30427 # multimin
#xe_lav_Ni = 0.49425
xe_lav_Nb = 0.30 + 1.0e-4 # landscape
xe_lav_Ni = 0.50 - 1.0e-4


g_gamma_raw = inVm * g_gamma.subs({FCC_A10CR: GAMMA_XCR,
                                   FCC_A10NB: GAMMA_XNB,
                                   FCC_A10NI: GAMMA_XNI,
                                   FCC_A11VA: 1.0,
                                   T: temp})

g_delta_raw = inVm * g_delta.subs({D0A_NBNI30NB: 4.0*DELTA_XNB,
                                   D0A_NBNI30NI: 1.0 - 4.0*DELTA_XNB,
                                   D0A_NBNI31CR: fr4by3 * DELTA_XCR,
                                   D0A_NBNI31NI: 1.0 - fr4by3 * DELTA_XCR,
                                   T: temp})

g_mu_raw = inVm * g_mu.subs({D85_NI7NB60NB: 1,
                             D85_NI7NB61CR: fr13by7*MU_XCR,
                             D85_NI7NB61NB: fr13by7*MU_XNB - fr6by7,
                             D85_NI7NB61NI: fr13by7*MU_XNI,
                             T: temp})

g_laves_raw = inVm * g_laves.subs({C14_LAVES0CR: 1.0 - fr3by2*LAVES_XNI,
                                   C14_LAVES0NI: fr3by2 * LAVES_XNI,
                                   C14_LAVES1CR: 1.0 - 3.0*LAVES_XNB,
                                   C14_LAVES1NB: 3.0 * LAVES_XNB,
                                   T: temp})

# Initialize parabolic curvatures from raw CALPHAD expressions
C_gam_Cr = diff(g_gamma_raw, GAMMA_XCR, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})
C_gam_Nb = diff(g_gamma_raw, GAMMA_XNB, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})

C_del_Cr = diff(g_delta_raw, DELTA_XCR, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})
C_del_Nb = diff(g_delta_raw, DELTA_XNB, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})

C_mu_Cr = diff(g_mu_raw, MU_XCR, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})
C_mu_Nb = diff(g_mu_raw, MU_XNB, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})

C_lav_Nb = diff(g_laves_raw, LAVES_XNB, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})
C_lav_Ni = diff(g_laves_raw, LAVES_XNI, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})

g0_gam = g_gamma_raw.subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})
g0_del = g_delta_raw.subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})
g0_mu =  g_mu_raw.subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})
g0_lav = g_laves_raw.subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})

print "Parabolic Gamma: %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_gam_Cr, xe_gam_Cr, C_gam_Nb, xe_gam_Nb, g0_gam)
print "Parabolic Delta: %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_del_Cr, xe_del_Cr, C_del_Nb, xe_del_Nb, g0_del)
print "Parabolic Mu:    %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_mu_Cr, xe_mu_Cr, C_mu_Nb, xe_mu_Nb, g0_mu)
print "Parabolic Laves: %.4e * (XNB - %.4f)**2 + %.4e * (XNI - %.4f)**2 + %.4e" % (C_lav_Nb, xe_lav_Nb, C_lav_Ni, xe_lav_Ni, g0_lav)
print ""

# Make substitutions
g_gamma = Piecewise((g_gamma_raw, Gt(GAMMA_XNI, 0) & Lt(GAMMA_XNI, 1) &
                                  Gt(GAMMA_XCR, 0) & Lt(GAMMA_XCR, 1) &
                                  Gt(GAMMA_XNB, 0) & Lt(GAMMA_XNB, 1)),
                    ( C_gam_Cr*(GAMMA_XCR - xe_gam_Cr)**2
                    + C_gam_Nb*(GAMMA_XNB - xe_gam_Nb)**2
                    + g0_gam, True))

g_delta = Piecewise((g_delta_raw, Le(DELTA_XCR, 0.75) & Le(DELTA_XNB, 0.25) &
                                  Gt(1-DELTA_XCR-DELTA_XNB, 0) & Lt(1-DELTA_XCR-DELTA_XNB, 1) &
                                  Gt(DELTA_XCR, 0) & Lt(DELTA_XCR, 1) &
                                  Gt(DELTA_XNB, 0) & Lt(DELTA_XNB, 1)),
                    ( C_del_Cr*(DELTA_XCR - xe_del_Cr)**2
                    + C_del_Nb*(DELTA_XNB - xe_del_Nb)**2
                    + g0_del, True))

g_mu    = Piecewise((g_mu_raw, Le(MU_XCR+MU_XNI, fr7by13) & Ge(MU_XNB, fr6by13) &
                               Gt(MU_XCR, 0) & Lt(MU_XCR, 1) &
                               Lt(MU_XNB, 1) & 
                               Gt(MU_XNI, 0) & Lt(MU_XNI, 1)),
                    ( C_mu_Cr*(MU_XCR - xe_mu_Cr)**2
                    + C_mu_Nb*(MU_XNB - xe_mu_Nb)**2
                    + g0_mu, True))

g_laves = Piecewise((g_laves_raw, Le(LAVES_XNB, fr1by3) & Le(LAVES_XNI, fr2by3) &
                                  Gt(LAVES_XNB, 0) & Gt(LAVES_XNI, 0) &
                                  Gt(1-LAVES_XNB-LAVES_XNI, 0) & Lt(1-LAVES_XNB-LAVES_XNI, 1)),
                    ( C_lav_Nb*(LAVES_XNB - xe_lav_Nb)**2   
                    + C_lav_Ni*(LAVES_XNI - xe_lav_Ni)**2
                    + g0_lav, True))


# Get free energy values at minima: should match, but won't if (x0,y0) are outside defined zone
g0_gam = g_gamma.subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})
g0_del = g_delta.subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})
g0_mu  = g_mu.subs( {MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})
g0_lav = g_laves.subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})


# Generate first derivatives
dGgam_dxCr = diff(g_gamma, GAMMA_XCR)
dGgam_dxNb = diff(g_gamma, GAMMA_XNB)
dGgam_dxNi = diff(g_gamma, GAMMA_XNI)

dGdel_dxCr = diff(g_delta, DELTA_XCR)
dGdel_dxNb = diff(g_delta, DELTA_XNB)

dGmu_dxCr = diff(g_mu, MU_XCR)
dGmu_dxNi = diff(g_mu, MU_XNI)
dGmu_dxNb = diff(g_mu, MU_XNB)

dGlav_dxNb = diff(g_laves, LAVES_XNB)
dGlav_dxNi = diff(g_laves, LAVES_XNI)
dGlavL_dxNb = diff(g_lavesLT, LAVES_XNB)
dGlavL_dxNi = diff(g_lavesLT, LAVES_XNI)

C_gam_Cr = diff(dGgam_dxCr, GAMMA_XCR).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})
C_gam_Nb = diff(dGgam_dxNb, GAMMA_XNB).subs({GAMMA_XCR: xe_gam_Cr, GAMMA_XNB: xe_gam_Nb, GAMMA_XNI: xe_gam_Ni})
print "Parabolic Gamma: %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_gam_Cr, xe_gam_Cr, C_gam_Nb, xe_gam_Nb, g0_gam)

C_del_Cr = diff(dGdel_dxCr, DELTA_XCR).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})
C_del_Nb = diff(dGdel_dxNb, DELTA_XNB).subs({DELTA_XCR: xe_del_Cr, DELTA_XNB: xe_del_Nb})
print "Parabolic Delta: %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_del_Cr, xe_del_Cr, C_del_Nb, xe_del_Nb, g0_del)

C_mu_Cr = diff(dGmu_dxCr, MU_XCR).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})
C_mu_Nb = diff(dGmu_dxNb, MU_XNB).subs({MU_XCR: xe_mu_Cr, MU_XNB: xe_mu_Nb, MU_XNI: xe_mu_Ni})
print "Parabolic Mu:    %.4e * (XCR - %.4f)**2 + %.4e * (XNB - %.4f)**2 + %.4e" % (C_mu_Cr, xe_mu_Cr, C_mu_Nb, xe_mu_Nb, g0_mu)

C_lav_Nb = diff(dGlav_dxNb, LAVES_XNB).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})
C_lav_Ni = diff(dGlav_dxNi, LAVES_XNI).subs({LAVES_XNB: xe_lav_Nb, LAVES_XNI: xe_lav_Ni})
print "Parabolic Laves: %.4e * (XNB - %.4f)**2 + %.4e * (XNI - %.4f)**2 + %.4e" % (C_lav_Nb, xe_lav_Nb, C_lav_Ni, xe_lav_Ni, g0_lav)


# Create parabolic approximation functions
g_gamma = C_gam_Cr * (GAMMA_XCR - xe_gam_Cr)**2 + C_gam_Nb * (GAMMA_XNB - xe_gam_Nb)**2 + g0_gam

g_delta = C_del_Cr * (DELTA_XCR - xe_del_Cr)**2 + C_del_Nb * (DELTA_XNB - xe_del_Nb)**2 + g0_del

g_mu    = C_mu_Cr  * (MU_XCR    - xe_mu_Cr )**2 + C_mu_Nb  * (MU_XNB    - xe_mu_Nb )**2 + g0_mu

g_laves = C_lav_Nb * (LAVES_XNB - xe_lav_Nb)**2 + C_lav_Ni * (LAVES_XNI - xe_lav_Ni)**2 + g0_lav


# Export C code
# Generate first derivatives
dGgam_dxCr = diff(g_gamma, GAMMA_XCR)
dGgam_dxNb = diff(g_gamma, GAMMA_XNB)
dGgam_dxNi = diff(g_gamma.subs({GAMMA_XNB: 1-GAMMA_XCR-GAMMA_XNI}), GAMMA_XNI)

dGdel_dxCr = diff(g_delta, DELTA_XCR)
dGdel_dxNb = diff(g_delta, DELTA_XNB)

dGmu_dxCr = diff(g_mu, MU_XCR)
#dGmu_dxNi = diff(g_mu, MU_XNI)
dGmu_dxNi = diff(g_mu.subs({MU_XNB: 1-MU_XCR-MU_XNI}), MU_XNI)
dGmu_dxNb = diff(g_mu, MU_XNB)

dGlavH_dxNb = diff(g_laves, LAVES_XNB)
dGlavH_dxNi = diff(g_laves, LAVES_XNI)
dGlavL_dxNb = diff(g_lavesLT, LAVES_XNB)
dGlavL_dxNi = diff(g_lavesLT, LAVES_XNI)


# Generate optimized second derivatives
d2Ggam_dxCrCr = diff(dGgam_dxCr, GAMMA_XCR)
d2Ggam_dxCrNb = diff(dGgam_dxCr, GAMMA_XNB)
d2Ggam_dxNbCr = diff(dGgam_dxNb, GAMMA_XCR)
d2Ggam_dxNbNb = diff(dGgam_dxNb, GAMMA_XNB)
d2Ggam_dxNiCr = diff(dGgam_dxNi, GAMMA_XCR)
d2Ggam_dxNiNb = diff(dGgam_dxNi, GAMMA_XNB)
d2Ggam_dxNiNi = diff(dGgam_dxNi, GAMMA_XNI)

d2Gdel_dxCrCr = diff(dGdel_dxCr, DELTA_XCR)
d2Gdel_dxCrNb = diff(dGdel_dxCr, DELTA_XNB)
d2Gdel_dxNbCr = diff(dGdel_dxNb, DELTA_XCR)
d2Gdel_dxNbNb = diff(dGdel_dxNb, DELTA_XNB)

d2Gmu_dxCrCr = diff(dGmu_dxCr, MU_XCR)
d2Gmu_dxCrNi = diff(dGmu_dxCr, MU_XNI)
d2Gmu_dxNiCr = diff(dGmu_dxNi, MU_XCR)
d2Gmu_dxNiNi = diff(dGmu_dxNi, MU_XNI)
d2Gmu_dxNbNb = diff(dGmu_dxNb, MU_XNB)

d2GlavH_dxNbNb = diff(dGlavH_dxNb, LAVES_XNB)
d2GlavH_dxNbNi = diff(dGlavH_dxNb, LAVES_XNI)
d2GlavH_dxNiNb = diff(dGlavH_dxNi, LAVES_XNB)
d2GlavH_dxNiNi = diff(dGlavH_dxNi, LAVES_XNI)

d2GlavL_dxNbNb = diff(dGlavL_dxNb, LAVES_XNB)
d2GlavL_dxNbNi = diff(dGlavL_dxNb, LAVES_XNI)
d2GlavL_dxNiNb = diff(dGlavL_dxNi, LAVES_XNB)
d2GlavL_dxNiNi = diff(dGlavL_dxNi, LAVES_XNI)

# Write Gibbs energy functions to disk, for direct use in phase-field code
codegen([# Gibbs energies
         ('g_gam',g_gamma), ('g_del',g_delta), ('g_mu',g_mu), ('g_lav',g_laves), ('g_lavLT',g_lavesLT),
         # First derivatives
         ('dg_gam_dxCr',dGgam_dxCr), ('dg_gam_dxNb',dGgam_dxNb), ('dg_gam_dxNi',dGgam_dxNi),
         ('dg_del_dxCr',dGdel_dxCr), ('dg_del_dxNb',dGdel_dxNb),
         ('dg_mu_dxCr',dGmu_dxCr), ('dg_mu_dxNb',dGmu_dxNb), ('dg_mu_dxNi',dGmu_dxNi),
         ('dg_lav_dxNb',dGlavH_dxNb), ('dg_lav_dxNi',dGlavH_dxNi),
         ('dg_lavLT_dxNb',dGlavL_dxNb), ('dg_lavLT_dxNi',dGlavL_dxNi),
         # Second derivatives
         ('d2g_gam_dxCrCr',  d2Ggam_dxCrCr), ('d2g_gam_dxCrNb',d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr',  d2Ggam_dxNbCr), ('d2g_gam_dxNbNb',d2Ggam_dxNbNb),
         ('d2g_gam_dxNiCr',  d2Ggam_dxNiCr), ('d2g_gam_dxNiNb',d2Ggam_dxNiNb),
         ('d2g_del_dxCrCr',  d2Gdel_dxCrCr), ('d2g_del_dxCrNb',d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr',  d2Gdel_dxNbCr), ('d2g_del_dxNbNb',d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',   d2Gmu_dxCrCr),  ('d2g_mu_dxCrNi', d2Gmu_dxCrNi),
         ('d2g_mu_dxNiCr',   d2Gmu_dxNiCr),  ('d2g_mu_dxNiNi', d2Gmu_dxNiNi),
         ('d2g_lav_dxNbNb',  d2GlavH_dxNbNb),('d2g_lav_dxNbNi',d2GlavH_dxNbNi),
         ('d2g_lav_dxNiNb',  d2GlavH_dxNiNb),('d2g_lav_dxNiNi',d2GlavH_dxNiNi),
         ('d2g_lavLT_dxNbNb',d2GlavL_dxNbNb),('d2g_lavLT_dxNbNi',d2GlavL_dxNbNi),
         ('d2g_lavLT_dxNiNb',d2GlavL_dxNiNb),('d2g_lavLT_dxNiNi',d2GlavL_dxNiNi)],
        language='C', prefix='parabola625', project='ALLOY625', to_files=True)


# Generate ternary axes

labels = [r'$\gamma$', r'$\delta$', r'$\mu$', 'LavesHT', 'LavesLT', 'BCC']
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

# triangle bounding the Gibbs simplex
XS = [0.0, 1.0, 0.5, 0.0]
YS = [0.0, 0.0,rt3by2, 0.0]
# triangle bounding three-phase coexistence
XT = [0.25, 0.4875+0.025/2,0.5375+0.4625/2, 0.25]
YT = [0.0,  0.025*rt3by2, 0.4625*rt3by2, 0.0]
# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05*i
    xni = 1.0 - xcr
    Xtick.append(xcr/2 - 0.002)
    Ytick.append(rt3by2*xcr)
    # Cr-Nb edge
    xcr = 0.05*i
    xnb = 1.0 - xcr
    Xtick.append(xnb + xcr/2 + 0.002)
    Ytick.append(rt3by2*xcr)


# Define threaded free energy calculator

def computeKernelExclusive(n):
    a = n / density # index along x-axis
    b = n % density # index along y-axis

    xnb = epsilon + 1.0*a / (density-1)
    xcr = epsilon + 1.0*b / (density-1)
    xni = 1.0 - xcr - xnb

    result = [0]*7
    
    if xni>0:
        result[0] = xcr
        result[1] = xnb
        result[2] = xni
        result[3] = g_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}) #Gg(xcr,xnb,xni)
        result[4] = g_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb}) #Gd(xcr,xnb)
        result[5] = g_mu.subs({MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}) #Gu(xcr,xnb,xni)
        result[6] = g_laves.subs({LAVES_XNB: xnb, LAVES_XNI: xni}) #Gh(xnb,xni)
        #result[7] = g_lavesLT.subs({LAVES_XNB: xnb, LAVES_XNI: xni}) #Gl(xnb,xni)
        #result[8] = g_bcc.subs({BCC_XCR: xcr, BCC_XNB: xnb, BCC_XNI: xni}) #Gb(xcr,xnb,xni)
    
    return result

# Generate ternary phase diagram

density = 501
allCr = []
allNb = []
allG = []
allID = []
points = []
phases = []

if __name__ == '__main__':
    starttime = time.time() # not exact, but multiprocessing makes time.clock() read from different cores

    #bar = FloatProgress(min=0,max=density**2)
    #display(bar)

    pool = Pool(6)

    i = 0
    for result in pool.imap(computeKernelExclusive, tqdm(range(density**2))):
        xcr, xnb, xni, fg, fd, fu, fh = result
        f = (fg, fd, fu, fh)

        # Accumulate (x, y, G) points for each phase
        if (fd**2 + fu**2 + fh**2) > epsilon:
            for n in range(len(f)):
                allCr.append(rt3by2*xcr)
                allNb.append(xnb + xcr/2)
                allG.append(f[n])
                allID.append(n)
        i += 1
        #bar.value = i

    pool.close()
    pool.join()
    
    points = np.array([allCr, allNb, allG]).T
    
    hull = ConvexHull(points)
    
    runtime = time.time() - starttime
    print "%ih:%im:%is elapsed" % (int(runtime/3600), int(runtime/60)%60, int(runtime)%60)


# Prepare arrays for plotting
X = [[],[],[],[], [], []]
Y = [[],[],[],[], [], []]
tielines = []

for simplex in hull.simplices:
    for i in simplex:
        X[allID[i]].append(allNb[i])
        Y[allID[i]].append(allCr[i])
        for j in simplex:
            if allID[i] != allID[j]:
                tielines.append([[allNb[i], allNb[j]], [allCr[i], allCr[j]]])


# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlim([0,1])
plt.ylim([0,rt3by2])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
n = 0
for tie in tielines:
    plt.plot(tie[0], tie[1], '-k', alpha=0.025)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=2.5, label=labels[i])
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
plt.legend(loc='best')
plt.savefig("parabolic_energy.png", bbox_inches='tight', dpi=400)

