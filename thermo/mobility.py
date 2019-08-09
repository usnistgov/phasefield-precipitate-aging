# -*- coding: utf-8 -*-
from math import exp, log
import numpy as np

from constants import *
from pyCinterface import *

def molfrac(wCr, wNb, wNi):
    nCr = wCr / 51.996
    nNb = wNb / 92.906
    nNi = wNi / 58.693
    N = nCr + nNb + nNi
    return (nCr/N, nNb/N, nNi/N)

wCr = xe2A()
wNb = xe1A()
wNi = 1 - wCr - wNb

xCr, xNb, xNi = molfrac(wCr, wNb, wNi)
print("xCr:", xCr, " xNb:", xNb, " xNi:", xNi)

# Mobility of (1) in pure (2), from `NIST-nifcc-mob.TDB`

M_Cr_Cr   = exp((-235000 - 82.0 * temp) / RT)
M_Cr_Nb   = exp((-287000 - 64.4 * temp) / RT)
M_Cr_Ni   = exp((-287000 - 64.4 * temp) / RT)
M_Cr_CrNi = exp((-68000)                / RT)

M_Nb_Cr   = exp((-255333 + RT * log(7.6071E-5)) / RT)
M_Nb_Nb   = exp((-274328 + RT * log(8.6440E-5)) / RT)
M_Nb_Ni   = exp((-255333 + RT * log(7.6071e-5)) / RT)

M_Ni_Cr   = exp((-235000 - 82.0 * temp)      / RT)
M_Ni_Nb   = exp((-287000 + RT * log(1.0E-4)) / RT)
M_Ni_Ni   = exp((-287000 - 69.8 * temp)      / RT)
M_Ni_CrNi = exp((-81000)                     / RT)

# === Atomic Mobilities in FCC Ni ===

M_Cr = xCr * M_Cr_Cr + xNb * M_Cr_Nb + xNi * M_Cr_Ni + xCr * xNi * M_Cr_CrNi
M_Nb = xCr * M_Nb_Cr + xNb * M_Nb_Nb + xNi * M_Nb_Ni
M_Ni = xCr * M_Ni_Cr + xNb * M_Ni_Nb + xNi * M_Ni_Ni + xCr * xNi * M_Ni_CrNi

print("M = ", np.array([M_Cr, M_Nb, M_Ni]))

# === Chemical Mobilities in FCC Ni ===

# k j    d_jCr-x_j   d_Crk-x_k                d_jNb-x_j   d_Nbk-x_k                d_jNi-x_j   d_Nik-x_k
M_CrCr = (1 - xCr) * (1 - xCr) * xCr * M_Cr + (   -xCr) * (   -xCr) * xNb * M_Nb + (   -xCr) * (   -xCr) * xNi * M_Ni
M_NbCr = (1 - xCr) * (   -xNb) * xCr * M_Cr + (   -xCr) * (1 - xNb) * xNb * M_Nb + (   -xCr) * (   -xNb) * xNi * M_Ni
M_CrNb = (   -xNb) * (1 - xCr) * xCr * M_Cr + (1 - xNb) * (   -xCr) * xNb * M_Nb + (   -xNb) * (   -xCr) * xNi * M_Ni
M_NbNb = (   -xNb) * (   -xNb) * xCr * M_Cr + (1 - xNb) * (1 - xNb) * xNb * M_Nb + (   -xNb) * (   -xNb) * xNi * M_Ni

# Second-derivatives of Gamma-phase free energy

dmu_CrCr = Vm * d2g_gam_dxCrCr()
dmu_CrNb = Vm * d2g_gam_dxCrNb()
dmu_NbCr = Vm * d2g_gam_dxNbCr()
dmu_NbNb = Vm * d2g_gam_dxNbNb()

# === Linear coefficients ===

# Pij = delta(i,j) - x_i
P = np.array([[1 - xCr,    -xCr],
              [   -xNb, 1 - xNb]])

M = np.array([[M_CrCr, M_CrNb],
              [M_NbCr, M_NbNb]])

print("M = ...\n", M)

L = P @ M @ P.transpose()

print("L = ...\n", L)
print("det(L) =", np.linalg.det(L))
print("λ =", np.linalg.eigvals(L))

# === D per Andersson & Ågren, Eq. 64, with only substitutional components (first summation)

# k j    dCrk-xk                  dmu_Crj    dNbk-xk                  dmu_Nbj
D_CrCr = (1 - xCr) * xCr * M_Cr * dmu_CrCr + (   -xCr) * xNb * M_Nb * dmu_NbCr
D_CrNb = (1 - xCr) * xCr * M_Cr * dmu_CrNb + (   -xCr) * xNb * M_Nb * dmu_NbNb
D_NbCr = (   -xNb) * xCr * M_Cr * dmu_CrCr + (1 - xNb) * xNb * M_Nb * dmu_NbCr
D_NbNb = (   -xNb) * xCr * M_Cr * dmu_CrNb + (1 - xNb) * xNb * M_Nb * dmu_NbNb

D = Vm * np.array([[D_CrCr, D_CrNb],
                   [D_NbCr, D_NbNb]])

print("D = ...\n", D)
print("det(D) =", np.linalg.det(D))
print("λ =", np.linalg.eigvals(D))


""" 
# ===  Output ===

xCr: 0.5585526948869939  xNb: 0.010717747618334031  xNi: 0.4307295574946722
M =  [1.88002933e-04 1.62333056e-16 4.78805808e-05]
M = ...
 [[ 2.68979789e-05 -3.73372284e-07]
 [-3.73372284e-07  1.44314915e-08]]
L = ...
 [[ 5.43039143e-06 -3.00530133e-07]
 [-3.00530133e-07  2.51312182e-08]]
det(L) = 4.615399119250355e-14
λ = [5.44704944e-06 8.47320953e-09]
D = ...
 [[ 2.06323441e-05  6.90879332e-05]
 [-5.00925602e-07 -1.67736222e-06]]
det(D) = 1.3906746028464745e-22
λ = [1.89549819e-05 7.33657587e-18]

"""

"""
# === L" per Andersson & Ågren ===
#     NB: returns M, not L"!
# k i    diCr - xi   dCrk - xk   xCr   M_Cr   diNb - xi   dNbk - xk   xNb   M_Nb   diNi - xi   dNik - xk   xNi   M_Ni
L_CrCr = (1 - xCr) * (1 - xCr) * xCr * M_Cr + (   -xCr) * (   -xCr) * xNb * M_Nb + (   -xCr) * (   -xCr) * xNi * M_Ni
L_CrNb = (   -xNb) * (1 - xCr) * xCr * M_Cr + (1 - xNb) * (   -xCr) * xNb * M_Nb + (   -xNb) * (   -xCr) * xNi * M_Ni
L_CrNi = (   -xNi) * (1 - xCr) * xCr * M_Cr + (   -xNi) * (   -xCr) * xNb * M_Nb + (1 - xNi) * (   -xCr) * xNi * M_Ni
L_NbCr = (1 - xCr) * (   -xNb) * xCr * M_Cr + (   -xCr) * (1 - xNb) * xNb * M_Nb + (   -xCr) * (   -xNb) * xNi * M_Ni
L_NbNb = (   -xNb) * (   -xNb) * xCr * M_Cr + (1 - xNb) * (1 - xNb) * xNb * M_Nb + (   -xNb) * (   -xNb) * xNi * M_Ni
L_NbNi = (   -xNi) * (   -xNb) * xCr * M_Cr + (   -xNi) * (1 - xNb) * xNb * M_Nb + (1 - xNi) * (   -xNb) * xNi * M_Ni
L_NiCr = (1  -xCr) * (   -xNi) * xCr * M_Cr + (   -xCr) * (   -xNi) * xNb * M_Nb + (   -xCr) * (1 - xNi) * xNi * M_Ni
L_NiNb = (   -xNb) * (   -xNi) * xCr * M_Cr + (1 - xNb) * (   -xNi) * xNb * M_Nb + (   -xNb) * (1 - xNi) * xNi * M_Ni
L_NiNi = (   -xNi) * (   -xNi) * xCr * M_Cr + (   -xNi) * (   -xNi) * xNb * M_Nb + (1 - xNi) * (1 - xNi) * xNi * M_Ni

L2 = np.array([[L_CrCr, L_CrNb, L_CrNi],
               [L_NbCr, L_NbNb, L_NbNi],
               [L_NiCr, L_NiNb, L_NiNi]])
print("L =", L2)
print("det(L) =", np.linalg.det(L2))
print("λ =", np.linalg.eigvals(L2))
"""
