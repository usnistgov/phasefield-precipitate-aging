# -*- coding: utf-8 -*-

from numpy import arange, sqrt

# Ni Alloy 625 particulars
temp = 870 + 273.15 # 1143 Kelvin
RT = 8.3144598*temp # J/mol/K
Vm = 1.0e-5         # m³/mol
inVm = 1.0 / Vm     # mol/m³

# Secondary-Phase Properties
s_delta = 0.155 # J/m²
s_laves = 0.205 # J/m²

# Specify gamma-delta-Laves corners (from phase diagram)
# with compositions as mass fractions

xe_gam_Cr = 0.5250
xe_gam_Nb = 0.0180

xe_del_Cr = 0.0258
xe_del_Nb = 0.2440

xe_lav_Cr = 0.3750
xe_lav_Nb = 0.2590

# allowable matrix compositions from ASTM F3056
matrixMinNb = 0.0202
matrixMaxNb = 0.0269
matrixMinCr = 0.2794
matrixMaxCr = 0.3288

# allowable enriched compositions centered on DICTRA
# (with same span as matrix)
enrichMinNb = 0.1659
enrichMaxNb = 0.1726
enrichMinCr = 0.2473
enrichMaxCr = 0.2967

# Let's avoid integer arithmetic in fractions.
fr3by4 = 0.75
fr3by2 = 1.5
fr4by3 = 4.0/3
fr2by3 = 2.0/3
fr1by4 = 0.25
fr1by3 = 1.0/3
fr1by2 = 0.5
rt3by2 = sqrt(3.0)/2
epsilon = 1e-10 # tolerance for comparing floating-point numbers to zero

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

# Triangular grid
XG = [[]]
YG = [[]]
for a in arange(0, 1, 0.1):
    # x1--x2: lines of constant x2=a
    XG.append([simX(a, 0), simX(a, 1-a)])
    YG.append([simY(0),    simY(1-a)])
    # x2--x3: lines of constant x3=a
    XG.append([simX(0, a), simX(1-a, a)])
    YG.append([simY(a),    simY(a)])
    # x1--x3: lines of constant x1=1-a
    XG.append([simX(0, a), simX(a, 0)])
    YG.append([simY(a),    simY(0)])
