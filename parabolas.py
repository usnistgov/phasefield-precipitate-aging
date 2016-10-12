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

# Make substitutions

#g_gamma = inVm * Piecewise((
#                        g_gamma.subs({FCC_A10CR: GAMMA_XCR,
#                        FCC_A10NB: GAMMA_XNB,
#                        FCC_A10NI: GAMMA_XNI,
#                        FCC_A11VA: 1.0,
#                        T: temp}),
#                        Gt(GAMMA_XCR, -epsilon) &
#                        Lt(GAMMA_XCR, 1+epsilon) &
#                        Gt(GAMMA_XNB, -epsilon) &
#                        Lt(GAMMA_XNB, 1+epsilon) &
#                        Gt(GAMMA_XNI, -epsilon) &
#                        Lt(GAMMA_XNI, 1+epsilon)),
#                        (  2.6e4*(GAMMA_XCR - 0.30)**2
#                         + 5.2e5*(GAMMA_XNB - 0.01)**2, True))

g_gamma   = inVm * (2.6e4*(GAMMA_XCR - 0.30)**2     + 5.2e5*(GAMMA_XNB - 0.01)**2 )
g_delta   = inVm * (1.5e6*(DELTA_XCR - 0.003125)**2 + 8.3e5*(DELTA_XNB - 0.24375 )**2 )
g_mu      = inVm * (9.5e4*(MU_XCR - 0.05  )**2      + 2.9e5*(1.0-MU_XCR-MU_XNI - 0.4875)**2 )
g_lavesHT = inVm * (8.2e5*(LAVES_XNB - 0.2875)**2   + 9.5e4*(LAVES_XNI - 0.3875)**2 )
g_lavesLT = inVm * (8.2e5*(LAVES_XNB - 0.2875)**2   + 9.5e4*(LAVES_XNI - 0.3875)**2)

#Ccr = 2.6e4
#Cnb = 5.2e5
#Cni = 9.5e4
#g_gamma = inVm * (  Ccr*(GAMMA_XCR - 0.30)**2     + Cnb*(GAMMA_XNB - 0.01)**2 )
#g_delta = inVm * (  Ccr*(DELTA_XCR - 0.003125)**2 + Cnb*(DELTA_XNB - 0.24375 )**2 )
#g_mu = inVm * (     Ccr*(MU_XCR - 0.05  )**2      + Cnb*(1.0-MU_XCR-MU_XNI - 0.4875)**2 )
#g_lavesHT = inVm * (Cnb*(LAVES_XNB - 0.2875)**2   + Cni*(LAVES_XNI - 0.3875)**2 )
#g_lavesLT = inVm * (Cnb*(LAVES_XNB - 0.2875)**2   + Cni*(LAVES_XNI - 0.3875)**2)


# Export C code
# Generate first derivatives
dGgam_dxCr = diff(g_gamma, GAMMA_XCR)
dGgam_dxNb = diff(g_gamma, GAMMA_XNB)
dGgam_dxNi = diff(g_gamma.subs({GAMMA_XCR: 1-GAMMA_XNB-GAMMA_XNI, GAMMA_XNB: 1-GAMMA_XCR-GAMMA_XNI},simultaneous=True), GAMMA_XNI)

dGdel_dxCr = diff(g_delta, DELTA_XCR)
dGdel_dxNb = diff(g_delta, DELTA_XNB)

dGmu_dxCr = diff(g_mu, MU_XCR)
dGmu_dxNi = diff(g_mu, MU_XNI)
dGmu_dxNb = diff(g_mu, MU_XNB)

dGlavH_dxNb = diff(g_lavesHT, LAVES_XNB)
dGlavH_dxNi = diff(g_lavesHT, LAVES_XNI)
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
         ('g_gam',g_gamma), ('g_mu',g_mu), ('g_lav',g_lavesHT), ('g_lavLT',g_lavesLT), ('g_del',g_delta),
         # First derivatives
         ('dg_gam_dxCr',dGgam_dxCr), ('dg_gam_dxNb',dGgam_dxNb), ('dg_gam_dxNi',dGgam_dxNi),
         ('dg_del_dxCr',dGdel_dxCr), ('dg_del_dxNb',dGdel_dxNb),
         ('dg_mu_dxCr',dGmu_dxCr), ('dg_mu_dxNi',dGmu_dxNi),
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
        result[6] = g_lavesHT.subs({LAVES_XNB: xnb, LAVES_XNI: xni}) #Gh(xnb,xni)
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

