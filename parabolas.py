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
from sympy import exp

# Visualization libraries
import matplotlib.pylab as plt

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
#xe_gam_Cr = 0.49 # phase diagram
#xe_gam_Nb = 0.03
xe_gam_Cr = 0.15
xe_gam_Nb = 0.0525
xe_gam_Ni = 1.0 - xe_gam_Cr - xe_gam_Nb

#xe_del_Cr = 0.02 # phase diagram
#xe_del_Nb = 0.225
xe_del_Cr = 0.0125
xe_del_Nb = 0.245

#xe_mu_Cr = 0.01 # phase diagram
#xe_mu_Nb = 0.61
xe_mu_Cr = 0.02
xe_mu_Nb = 0.5
xe_mu_Ni = 1.0 - xe_mu_Cr - xe_mu_Nb

#xe_lav_Nb = 0.32 # phase diagram
#xe_lav_Ni = 0.34
xe_lav_Nb = 0.29
xe_lav_Ni = 0.37



X0 = (simX(xe_gam_Nb, xe_gam_Cr), simX(xe_del_Nb, xe_del_Cr), simX(xe_mu_Nb, xe_mu_Cr), simX(xe_lav_Nb, 1-xe_lav_Nb-xe_lav_Ni))
Y0 = (simY(xe_gam_Cr), simY(xe_del_Cr), simY(xe_mu_Cr), simY(1-xe_lav_Nb-xe_lav_Ni))

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


print "Parabolic Gamma: ", p_gamma
print "Parabolic Delta: ", p_delta
print "Parabolic Mu:    ", p_mu
print "Parabolic Laves: ", p_laves
print ""

# Export C code
# Generate first derivatives
dGgam_dxCr = diff(p_gamma, GAMMA_XCR)
dGgam_dxNb = diff(p_gamma, GAMMA_XNB)
dGgam_dxNi = diff(p_gamma, GAMMA_XNI)

dGdel_dxCr = diff(p_delta, DELTA_XCR)
dGdel_dxNb = diff(p_delta, DELTA_XNB)

dGmu_dxCr = diff(p_mu, MU_XCR)
dGmu_dxNb = diff(p_mu, MU_XNB)

dGlav_dxCr = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
dGlav_dxNb = diff(p_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)


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
         ('g_gam',p_gamma), ('g_del',p_delta), ('g_mu',p_mu), ('g_lav',p_laves),
         # Constants
         ('xe_gam_Cr',xe_gam_Cr), ('xe_gam_Nb',xe_gam_Nb),
         ('xe_del_Cr',xe_del_Cr), ('xe_del_Nb',xe_del_Nb),
         ('xe_mu_Cr', xe_mu_Cr),  ('xe_mu_Nb', xe_mu_Nb),
         ('xe_lav_Nb',xe_lav_Nb), ('xe_lav_Ni',xe_lav_Ni),
         # First derivatives
         ('dg_gam_dxCr',dGgam_dxCr), ('dg_gam_dxNb',dGgam_dxNb), ('dg_gam_dxNi',dGgam_dxNi),
         ('dg_del_dxCr',dGdel_dxCr), ('dg_del_dxNb',dGdel_dxNb),
         ('dg_mu_dxCr', dGmu_dxCr),  ('dg_mu_dxNb', dGmu_dxNb),
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



# Create Taylor approximation functions
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


print "Taylor Gamma: ", t_gamma
print "Taylor Delta: ", t_delta
print "Taylor Mu:    ", t_mu
print "Taylor Laves: ", t_laves
print ""

# Export C code
# Generate first derivatives
dGgam_dxCr = diff(t_gamma, GAMMA_XCR)
dGgam_dxNb = diff(t_gamma, GAMMA_XNB)
dGgam_dxNi = diff(t_gamma, GAMMA_XNI)

dGdel_dxCr = diff(t_delta, DELTA_XCR)
dGdel_dxNb = diff(t_delta, DELTA_XNB)

dGmu_dxCr = diff(t_mu, MU_XCR)
dGmu_dxNb = diff(t_mu, MU_XNB)

dGlav_dxCr = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XCR)
dGlav_dxNb = diff(t_laves.subs({LAVES_XNI: 1.0-LAVES_XCR-LAVES_XNB}), LAVES_XNB)


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
         ('g_gam',t_gamma), ('g_del',t_delta), ('g_mu',t_mu), ('g_lav',t_laves),
         # Constants
         ('xe_gam_Cr',xe_gam_Cr), ('xe_gam_Nb',xe_gam_Nb),
         ('xe_del_Cr',xe_del_Cr), ('xe_del_Nb',xe_del_Nb),
         ('xe_mu_Cr', xe_mu_Cr),  ('xe_mu_Nb', xe_mu_Nb),
         ('xe_lav_Nb',xe_lav_Nb), ('xe_lav_Ni',xe_lav_Ni),
         # First derivatives
         ('dg_gam_dxCr',dGgam_dxCr), ('dg_gam_dxNb',dGgam_dxNb), ('dg_gam_dxNi',dGgam_dxNi),
         ('dg_del_dxCr',dGdel_dxCr), ('dg_del_dxNb',dGdel_dxNb),
         ('dg_mu_dxCr', dGmu_dxCr),  ('dg_mu_dxNb', dGmu_dxNb),
         ('dg_lav_dxCr',dGlav_dxCr), ('dg_lav_dxNb',dGlav_dxNb),
         # Second derivatives
         ('d2g_gam_dxCrCr',  d2Ggam_dxCrCr), ('d2g_gam_dxCrNb',d2Ggam_dxCrNb),
         ('d2g_gam_dxNbCr',  d2Ggam_dxNbCr), ('d2g_gam_dxNbNb',d2Ggam_dxNbNb),
         ('d2g_del_dxCrCr',  d2Gdel_dxCrCr), ('d2g_del_dxCrNb',d2Gdel_dxCrNb),
         ('d2g_del_dxNbCr',  d2Gdel_dxNbCr), ('d2g_del_dxNbNb',d2Gdel_dxNbNb),
         ('d2g_mu_dxCrCr',   d2Gmu_dxCrCr),  ('d2g_mu_dxCrNb', d2Gmu_dxCrNb),
         ('d2g_mu_dxNbCr',   d2Gmu_dxNbCr),  ('d2g_mu_dxNbNb', d2Gmu_dxNbNb),
         ('d2g_lav_dxCrCr',  d2Glav_dxCrCr), ('d2g_lav_dxCrNb',d2Glav_dxCrNb),
         ('d2g_lav_dxNbCr',  d2Glav_dxNbCr), ('d2g_lav_dxNbNb',d2Glav_dxNbNb)],
        language='C', prefix='taylor625', project='ALLOY625', to_files=True)



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

density = 101

allY = []
allX = []
allG = []
allID = []
points = []
phases = []

#for xcr in np.linspace(0, 1, density, endpoint=False):
#    for xnb in np.linspace(0, 1, density, endpoint=False):
for phi in np.linspace(-1, 1, density): #, endpoint=False):
    for psi in np.linspace(0, 1, density, endpoint=False):
        xni = (1+phi)*(1-psi)/2 + 0.5*(np.random.random_sample() - 0.5)/(density-1)
        xnb = (1-phi)*(1-psi)/2 + 0.5*(np.random.random_sample() - 0.5)/(density-1)
        xcr = 1 - xni - xnb # psi + (np.random.random_sample() - 0.5)/(density-1)
        if (xcr < 1  and xcr > 0) \
        and (xnb < 1  and xnb > 0) \
        and (xni < 1  and xni > 0):
            f = (t_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}),
                 t_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}),
                 t_mu.subs(   {MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}),
                 t_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNB: xnb, LAVES_XNI: xni}))
            for n in range(len(f)):
                allX.append(xnb + xcr/2)
                allY.append(rt3by2*xcr)
                allG.append(f[n])
                allID.append(n)

points = np.array([allX, allY, allG]).T
    
hull = ConvexHull(points)
    

# Prepare arrays for plotting
X = [[],[],[],[]]
Y = [[],[],[],[]]
tielines = []

for simplex in hull.simplices:
    if len(simplex) != 3:
        print simplex
    for i in simplex:
        X[allID[i]].append(allX[i])
        Y[allID[i]].append(allY[i])
        for j in simplex:
            if j>i and allID[i] != allID[j]:
                tielines.append([[allX[i], allX[j]], [allY[i], allY[j]]])


# Plot phase diagram
pltsize = 20
plt.figure(figsize=(pltsize, rt3by2*pltsize))
plt.title("Cr-Nb-Ni at %.0fK"%temp, fontsize=18)
plt.xlim([0,1])
plt.ylim([0,rt3by2])
plt.xlabel(r'$x_\mathrm{Nb}$', fontsize=24)
plt.ylabel(r'$x_\mathrm{Cr}$', fontsize=24)
plt.plot(XS, YS, '-k')
for tie in tielines:
    plt.plot(tie[0], tie[1], '-k', alpha=0.025)
for i in range(len(labels)):
    plt.scatter(X[i], Y[i], color=colors[i], s=2.5, label=labels[i])
plt.scatter(X0, Y0, color='black', s=5)
plt.xticks(np.linspace(0, 1, 21))
plt.scatter(Xtick, Ytick, color='black', s=3)
#plt.scatter(0.02+0.3/2, rt3by2*0.3, color='red', s=8)
plt.legend(loc='best')
plt.savefig("parabolic_energy.png", bbox_inches='tight', dpi=100)
plt.close()



# Compare CALPHAD and parabolic expressions (const. Cr)
stepsz = 0.01

# Plot phase diagram
plt.figure()
plt.title("Cr-Nb-Ni at %.0fK"%temp)
plt.xlabel(r'$x_\mathrm{Nb}$')
#plt.xlabel(r'$x_\mathrm{Cr}$')
plt.ylabel(r'$\mathcal{F}$')
#plt.xlim([0, 0.625])
plt.ylim([-1e10, 0])

#for xcr in (0.01, 0.1, 0.2, 0.3, 0.4, 0.5):
for xcr in (0.1, 0.2):
#for xcr in (0.3, 0.4):
#for xnb in (0.01, 0.05, 0.1, 0.15, 0.2, 0.25):
    xgam = []
    cgam = []
    tgam = []
    tdel = []
    tmu  = []
    tlav = []
    for xnb in np.arange(0.01, 0.98, stepsz):
    #for xcr in np.arange(0.01, 0.6, stepsz):
        xni = 1.0 - xcr - xnb
        xgam.append(xnb)
        #xgam.append(xcr)
    #    cgam.append(g_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}))
    #    tgam.append(t_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni}))
    #    tdel.append(t_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb}))
        tmu.append( t_mu.subs({MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}))
    #    tlav.append(t_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNI: xni}))

    #xdel = []
    #cdel = []
    #for xnb in np.arange(0.01, 0.25, stepsz):
    ##for xcr in np.arange(0.01, 0.75, stepsz):
    #    xni = 1.0 - xcr - xnb
    #    xdel.append(xnb)
    ##    xdel.append(xcr)
    #    cdel.append(g_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb, DELTA_XNI: xni}))

    xmu  = []
    cmu = []
    for xnb in np.arange(6.0/13, 0.98, stepsz):
    #for xcr in np.arange(0.01, 7.0/13, stepsz):
        xni = 1.0 - xcr - xnb
        xmu.append(xnb)
    #    xmu.append(xcr)
        cmu.append(g_mu.subs({MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni}))

    #xlav = []
    #clav = []
    #for xnb in np.arange(0.01, 0.33, stepsz):
    ##for xcr in np.arange(0.01, 0.67, stepsz):
    #    xni = 1.0 - xcr - xnb
    #    xlav.append(xnb)
    ##    xlav.append(xcr)
    #    clav.append(g_laves.subs({LAVES_XCR: xcr, LAVES_XNB: xnb, LAVES_XNI: xni}))

    #plt.plot(xgam, cgam, label=r'$\gamma$ CALPHAD')
    #plt.plot(xgam, tgam, label=r'$\gamma$ Taylor')

    #plt.plot(xdel, cdel, label=r'$\delta$ CALPHAD')
    #plt.plot(xgam, tdel, label=r'$\delta$ Taylor')

    plt.plot(xmu,  cmu,  label=r'$\mu$ CALPHAD')
    plt.plot(xgam, tmu,  label=r'$\mu$ Taylor')

    #plt.plot(xlav, clav, label=r'L CALPHAD')
    #plt.plot(xgam, tlav, label=r'L Taylor')

plt.legend(loc='best', fontsize=6)
plt.savefig("linescan.png", bbox_inches='tight', dpi=300)





## Make substitutions
#g_gamma = Piecewise((g_gamma, Gt(GAMMA_XNI, 0) & Lt(GAMMA_XNI, 1) &
#                              Gt(GAMMA_XCR, 0) & Lt(GAMMA_XCR, 1) &
#                              Gt(GAMMA_XNB, 0) & Lt(GAMMA_XNB, 1)),
#                    (t_gamma, True))
#
#g_delta = Piecewise((g_delta, Le(DELTA_XCR, 0.75) & Le(DELTA_XNB, 0.25) &
#                              Gt(1-DELTA_XCR-DELTA_XNB, 0) & Lt(1-DELTA_XCR-DELTA_XNB, 1) &
#                              Gt(DELTA_XCR, 0) & Lt(DELTA_XCR, 1) &
#                              Gt(DELTA_XNB, 0) & Lt(DELTA_XNB, 1)),
#                    (t_delta, True))
#
#g_mu    = Piecewise((g_mu, Le(MU_XCR+MU_XNI, fr7by13) & Ge(MU_XNB, fr6by13) &
#                           Gt(MU_XCR, 0) & Lt(MU_XCR, 1) &
#                           Lt(MU_XNB, 1) & 
#                           Gt(MU_XNI, 0) & Lt(MU_XNI, 1)),
#                    (t_mu, True))
#
#g_laves = Piecewise((g_laves, Le(LAVES_XNB, fr1by3) & Le(LAVES_XNI, fr2by3) &
#                              Gt(LAVES_XNB, 0) & Gt(LAVES_XNI, 0) &
#                              Gt(1-LAVES_XNB-LAVES_XNI, 0) & Lt(1-LAVES_XNB-LAVES_XNI, 1)),
#                    (t_laves, True))


