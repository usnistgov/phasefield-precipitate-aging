#!/usr/bin/python

# Numerical libraries
import numpy as np
from scipy.optimize import fsolve
from sympy.utilities.lambdify import lambdify
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

# Visualization libraries
import matplotlib.pylab as plt

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
xe_del_Cr = 0.0088 # landscape
xe_del_Nb = 0.2493

#xe_mu_Cr = 0.025 # original
#xe_mu_Nb = 0.4875
#xe_mu_Cr = 0.02792 # multimin
#xe_mu_Nb = 0.49076
xe_mu_Cr = 0.0106 # landscape
xe_mu_Nb = 0.5084
xe_mu_Ni = 1.0 - xe_mu_Cr - xe_mu_Nb

#xe_lav_Nb = 0.2625 # original
#xe_lav_Ni = 0.375
#xe_lav_Nb = 0.30427 # multimin
#xe_lav_Ni = 0.49425
xe_lav_Nb = 0.306 # landscape
xe_lav_Ni = 0.491


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

dGdel_dxCr = diff(g_delta, DELTA_XCR)
dGdel_dxNb = diff(g_delta, DELTA_XNB)

dGmu_dxCr = diff(g_mu, MU_XCR)
dGmu_dxNb = diff(g_mu, MU_XNB)

dGlav_dxCr = diff(g_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
dGlav_dxNb = diff(g_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)


# Generate optimized second derivatives
d2Ggam_dxCrCr = diff(dGgam_dxCr, GAMMA_XCR)
d2Ggam_dxCrNb = diff(dGgam_dxCr, GAMMA_XNB)
d2Ggam_dxNbCr = diff(dGgam_dxNb, GAMMA_XCR)
d2Ggam_dxNbNb = diff(dGgam_dxNb, GAMMA_XNB)

d2Gdel_dxCrCr = diff(dGdel_dxCr, DELTA_XCR)
d2Gdel_dxCrNb = diff(dGdel_dxCr, DELTA_XNB)
d2Gdel_dxNbCr = diff(dGdel_dxNb, DELTA_XCR)
d2Gdel_dxNbNb = diff(dGdel_dxNb, DELTA_XNB)

d2Gmu_dxCrCr = diff(dGmu_dxCr, MU_XCR)
d2Gmu_dxCrNb = diff(dGmu_dxCr, MU_XNI)
d2Gmu_dxNbCr = diff(dGmu_dxNb, MU_XCR)
d2Gmu_dxNbNb = diff(dGmu_dxNb, MU_XNB)

d2Glav_dxCrCr = diff(dGlav_dxCr, LAVES_XNB)
d2Glav_dxCrNb = diff(dGlav_dxCr, LAVES_XNB)
d2Glav_dxNbCr = diff(dGlav_dxNb, LAVES_XCR)
d2Glav_dxNbNb = diff(dGlav_dxNb, LAVES_XNB)

# Write Gibbs energy functions to disk, for direct use in phase-field code
codegen([# Gibbs energies
         ('g_gam',g_gamma), ('g_del',g_delta), ('g_mu',g_mu), ('g_lav',g_laves), ('g_lavLT',g_lavesLT),
         # First derivatives
         ('dg_gam_dxCr',dGgam_dxCr), ('dg_gam_dxNb',dGgam_dxNb), ('dg_gam_dxNi',dGgam_dxNi),
         ('dg_del_dxCr',dGdel_dxCr), ('dg_del_dxNb',dGdel_dxNb),
         ('dg_mu_dxCr',dGmu_dxCr), ('dg_mu_dxNb',dGmu_dxNb), ('dg_mu_dxNi',dGmu_dxNi),
         ('dg_lav_dxCr',dGlav_dxCr), ('dg_lav_dxNb',dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr',  d2Ggam_dxCrCr), ('d2g_gam_dxCrNb',d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr',  d2Ggam_dxNbCr), ('d2g_gam_dxNbNb',d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr',  d2Gdel_dxCrCr), ('d2g_del_dxCrNb',d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr',  d2Gdel_dxNbCr), ('d2g_del_dxNbNb',d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',   d2Gmu_dxCrCr),  ('d2g_mu_dxCrNb', d2Gmu_dxCrNb),
         ('d2g_mu_dxNbCr',   d2Gmu_dxNbCr),  ('d2g_mu_dxNbNb', d2Gmu_dxNbNb),
         ('d2g_lav_dxCrCr',  d2Glav_dxCrCr),('d2g_lav_dxCrNb',d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr',  d2Glav_dxNbCr),('d2g_lav_dxNbNb',d2Glav_dxNbNb)],
        language='C', prefix='parabola625', project='ALLOY625', to_files=True)



# Define parabolic functions (faster than sympy)
def pg(xcr, xnb):
    return C_gam_Cr * (xcr - xe_gam_Cr)**2 + C_gam_Nb * (xnb - xe_gam_Nb)**2 + g0_gam
def pd(xcr, xnb):
    return C_del_Cr * (xcr - xe_del_Cr)**2 + C_del_Nb * (xnb - xe_del_Nb)**2 + g0_del
def pm(xcr, xnb):
    return C_mu_Cr  * (xcr - xe_mu_Cr )**2 + C_mu_Nb  * (xnb - xe_mu_Nb )**2 + g0_mu
def pl(xnb, xni):
    return C_lav_Nb * (xnb - xe_lav_Nb)**2 + C_lav_Ni * (xni - xe_lav_Ni)**2 + g0_lav


# Generate ternary axes
labels = [r'$\gamma$', r'$\delta$', r'$\mu$', 'Laves']
colors = ['red', 'green', 'blue', 'cyan']

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



# Generate ternary phase diagram

density = 1001
allCr = []
allNb = []
allG = []
allID = []
points = []
phases = []

for xcr in np.linspace(epsilon, 1.0-epsilon, num=density):
    for xnb in np.linspace(epsilon, 1.0-epsilon, num=density):
        xni = 1.0 - xcr - xnb
        if xni > -epsilon: # and xni < 1.0+epsilon:
            f = (pg(xcr, xnb), pd(xcr, xnb), pm(xcr, xnb), pl(xnb, xni))
            for n in range(len(f)):
                allCr.append(rt3by2*xcr)
                allNb.append(xnb + xcr/2)
                allG.append(f[n])
                allID.append(n)
    
points = np.array([allCr, allNb, allG]).T
    
hull = ConvexHull(points)
    

# Prepare arrays for plotting
X = [[],[],[],[]]
Y = [[],[],[],[]]
tielines = []

for simplex in hull.simplices:
    for i in simplex:
        X[allID[i]].append(allNb[i])
        Y[allID[i]].append(allCr[i])
        #for j in simplex:
        #    if allID[i] != allID[j]:
        #        tielines.append([[allNb[i], allNb[j]], [allCr[i], allCr[j]]])


# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlim([0,1])
plt.ylim([0,rt3by2])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
#for tie in tielines:
#    plt.plot(tie[0], tie[1], '-k', alpha=0.025)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=2.5, label=labels[i])
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
#plt.scatter(0.02+0.3/2, rt3by2*0.3, color='red', s=8)
plt.legend(loc='best')
plt.savefig("parabolic_energy.png", bbox_inches='tight', dpi=400)

