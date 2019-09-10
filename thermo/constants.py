# -*- coding: utf-8 -*-

from numpy import arange, sqrt

def molfrac(wCr, wNb, wNi):
    # Assume 1 g of material
    nCr = wCr / 51.996 # g / (g/mol) = mol
    nNb = wNb / 92.906
    nNi = wNi / 58.693
    nTot = nCr + nNb + nNi
    return (nCr / nTot, nNb / nTot, nNi / nTot)

def wt_frac(xCr, xNb, xNi):
    # Assume 1 mol of material
    mCr = xCr * 51.996 # mol * (g/mol) = g
    mNb = xNb * 92.906
    mNi = xNi * 58.693
    mTot = mCr + mNb + mNi
    return (mCr / mTot, mNb / mTot, mNi / mTot)

# Ni Alloy 625 particulars
temp = 870 + 273.15  # 1143 K
RT = 8.31446261815324 * temp  # J/mol/K
Vm = 1.0e-5  # m³/mol
inVm = 1.0 / Vm  # mol/m³

# Secondary-Phase Properties
s_delta = 0.13 # 0.079  # J/m²
s_laves = s_delta  # J/m²

# Specify gamma-delta-Laves corners (from phase diagram)
# with compositions as mass fractions

we_gam_Cr = 0.5250
we_gam_Nb = 0.0180
we_gam_Ni = 1 - we_gam_Cr - we_gam_Nb

we_del_Cr = 0.0258
we_del_Nb = 0.2440
we_del_Ni = 1 - we_del_Cr - we_del_Nb

we_lav_Cr = 0.3750
we_lav_Nb = 0.2590
we_lav_Ni = 1 - we_lav_Cr - we_lav_Nb

xe_gam_Cr, xe_gam_Nb, xe_gam_Ni = molfrac(we_gam_Cr, we_gam_Nb, 1 - we_gam_Cr - we_gam_Nb)
xe_del_Cr, xe_del_Nb, xe_del_Ni = molfrac(we_del_Cr, we_del_Nb, 1 - we_del_Cr - we_del_Nb)
xe_lav_Cr, xe_lav_Nb, xe_lav_Ni = molfrac(we_lav_Cr, we_lav_Nb, 1 - we_lav_Cr - we_lav_Nb)

# allowable matrix compositions (mass fractions), from ASTM F3056 (TKR5p238)
matrixMinNb_w = 0.0315
matrixMaxNb_w = 0.0415
matrixMinCr_w = 0.2800
matrixMaxCr_w = 0.3300

# allowable matrix compositions (mole fractions)

matrixMinCr, matrixMinNb, matrixMinNi = molfrac(matrixMinCr_w, matrixMinNb_w, 1 - matrixMinCr_w - matrixMinNb_w)
matrixMaxCr, matrixMaxNb, matrixMaxNi = molfrac(matrixMaxCr_w, matrixMaxNb_w, 1 - matrixMaxCr_w - matrixMaxNb_w)

# allowable enriched compositions (mass fractions) centered on DICTRA
# (with same span as matrix)
enrichMinNb_w = 0.235 - (matrixMaxNb_w - matrixMinNb_w) / 2
enrichMaxNb_w = 0.235 + (matrixMaxNb_w - matrixMinNb_w) / 2
enrichMinCr_w = 0.275 - (matrixMaxCr_w - matrixMinCr_w) / 2
enrichMaxCr_w = 0.275 + (matrixMaxCr_w - matrixMinCr_w) / 2

# allowable enriched compositions (mole fractions)

enrichMinCr, enrichMinNb, enrichMinNi = molfrac(enrichMinCr_w, enrichMinNb_w, 1 - enrichMinCr_w - enrichMinNb_w)
enrichMaxCr, enrichMaxNb, enrichMaxNi = molfrac(enrichMaxCr_w, enrichMaxNb_w, 1 - enrichMaxCr_w - enrichMaxNb_w)

# Let's avoid integer arithmetic in fractions.
fr3by4 = 0.75
fr3by2 = 1.5
fr4by3 = 4.0 / 3
fr2by3 = 2.0 / 3
fr1by4 = 0.25
fr1by3 = 1.0 / 3
fr1by2 = 0.5
rt3by2 = sqrt(3.0) / 2
epsilon = 1e-10  # tolerance for comparing floating-point numbers to zero

# Helper functions to convert compositions into (x,y) coordinates
def simX(x2, x3):
    return x2 + fr1by2 * x3

def simY(x3):
    return rt3by2 * x3

# triangle bounding the Gibbs simplex
XS = [0, simX(1, 0), simX(0, 1), 0]
YS = [0, simY(0), simY(1), 0]

# Tick marks along simplex edges
Xtick = []
Ytick = []
for i in range(20):
    # Cr-Ni edge
    xcr = 0.05 * i
    xnb = -0.002
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))
    # Cr-Nb edge
    xcr = 0.05 * i
    xnb = 1.002 - xcr
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))
    # Nb-Ni edge
    xcr = -0.002
    xnb = 0.05 * i
    Xtick.append(simX(xnb, xcr))
    Ytick.append(simY(xcr))

# Triangular grid
XG = [[]]
YG = [[]]
for a in arange(0, 1, 0.1):
    # x1--x2: lines of constant x2=a
    XG.append([simX(a, 0), simX(a, 1 - a)])
    YG.append([simY(0), simY(1 - a)])
    # x2--x3: lines of constant x3=a
    XG.append([simX(0, a), simX(1 - a, a)])
    YG.append([simY(a), simY(a)])
    # x1--x3: lines of constant x1=1-a
    XG.append([simX(0, a), simX(a, 0)])
    YG.append([simY(a), simY(0)])
