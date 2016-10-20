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
from matplotlib.colors import LogNorm


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
xe_gam_Cr = 1.0e-2 # landscape
xe_gam_Nb = 1.0/3 - 1.0e-2
xe_gam_Ni = 1.0 - xe_gam_Cr - xe_gam_Nb

xe_del_Cr = 0.0088 # landscape
xe_del_Nb = 0.2493

xe_mu_Cr = 0.0106 # landscape
xe_mu_Nb = 0.5084      
xe_mu_Ni = 1.0 - xe_mu_Cr - xe_mu_Nb

xe_lav_Nb = 0.306 # landscape
xe_lav_Ni = 0.491


# Substitute lattice for system variables
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

g_mu_raw    = inVm * g_mu.subs({D85_NI7NB60NB: 1.0,
                                D85_NI7NB61CR: fr13by7*MU_XCR,
                                D85_NI7NB61NB: fr13by7*MU_XNB - fr6by7,
                                D85_NI7NB61NI: fr13by7*MU_XNI,
                                T: temp})

g_laves_raw = inVm * g_laves.subs({C14_LAVES0CR: 1.0 - fr3by2*LAVES_XNI,
                                   C14_LAVES0NI: fr3by2 * LAVES_XNI,
                                   C14_LAVES1CR: 1.0 - 3.0*LAVES_XNB,
                                   C14_LAVES1NB: 3.0 * LAVES_XNB,
                                   T: temp})


# Specify parabolic approximations from raw CALPHAD expressions
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


# Build piecewise expressions
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
                               Gt(MU_XNB, 0) & Lt(MU_XNB, 1) &
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

#fmax = g_laves.subs({LAVES_XNB: 1.0 - 1.0e-4, LAVES_XNI: 1.0e-4}) # 1.144e+11
#fmax = 5.0e10
#print "Energy ceiling in phase graphics is %.4g J/m**3." % (fmax)

# Plot ternary free energy landscapes
Titles = (r'$\gamma$', r'$\delta$', r'$\mu$', r'Laves')
npts = 200
nfun = 4
#levels = 100
#levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100, 1000, 10000, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
span = (-0.05,1.05)
yspan = (-0.15,0.95)
x = np.linspace(span[0],span[1],npts)
y = np.linspace(yspan[0],yspan[1],npts)
z = np.ndarray(shape=(nfun,len(x)*len(y)), dtype=float)

trix = np.array([0,1,0.5,0])
triy = np.array([0,0,np.sqrt(3)/2,0])
offx = 0.5*(trix - 1)
offy = 0.5*(triy - 1)

p = np.zeros(len(x)*len(y))
q = np.zeros(len(x)*len(y))

n=0
for j in tqdm(np.nditer(y)):
    for i in np.nditer(x):
        xcr = 2.0 * j / np.sqrt(3.0)
        xnb = i - j / np.sqrt(3.0)
        xni = 1.0 - xcr - xnb
        p[n]=i
        q[n]=j
        z[0][n] = g_gamma.subs({GAMMA_XCR: xcr, GAMMA_XNB: xnb, GAMMA_XNI: xni})
        z[1][n] = g_delta.subs({DELTA_XCR: xcr, DELTA_XNB: xnb})
        z[2][n] = g_mu.subs( {MU_XCR: xcr, MU_XNB: xnb, MU_XNI: xni})
        z[3][n] = g_laves.subs({LAVES_XNB: xnb, LAVES_XNI: xni})
        n+=1

datmin = 1e13
datmax = 1e-13
for n in range(len(z)):
    mymin = np.min(z[n])
    mymax = np.max(z[n])
    datmin = min(datmin, mymin)
    datmax = max(datmax, mymax)

print "Data spans [%.4g, %.4g]" % (datmin, datmax)

levels = np.logspace(np.log2(datmin-1.0001*datmin), np.log2(datmax-1.01*datmin), num=40, base=2.0)

#print "Contour levels: ", levels

f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
f.suptitle("IN625 Ternary Potentials",fontsize=14)
n=0
for ax in axarr.reshape(-1):
    ax.set_title(Titles[n],fontsize=10)
    ax.axis('equal')
    ax.set_xlim(span)
    ax.set_ylim(yspan)
    ax.axis('off')
    ax.tricontourf(p, q, z[n]-1.01*np.min(z[n]), levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    ax.plot(trix,triy, linestyle=':', color='w')
    n+=1
plt.figtext(x=0.5, y=0.0625, ha='center', fontsize=8, \
            s=r'White triangles enclose Gibbs simplex, $x_{\mathrm{Cr}}+x_{\mathrm{Nb}}+x_{\mathrm{Ni}}=1$.')
f.savefig('ternary.png', dpi=600, bbox_inches='tight')
plt.close()

files = ['diagrams/parabola_gamma.png', 'diagrams/parabola_delta.png', 'diagrams/parabola_mu.png', 'diagrams/parabola_Laves.png']
for n in range(len(z)):
    plt.axis('equal')
    plt.xlim(span)
    plt.ylim(yspan)
    plt.axis('off')
    plt.tricontourf(p, q, z[n]-1.01*np.min(z[n]), levels, cmap=plt.cm.get_cmap('coolwarm'), norm=LogNorm())
    plt.plot(trix,triy, linestyle=':', color='w')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(files[n], transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

