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

# Second-derivatives of Gamma-phase free energy

dmu_CrCr = Vm * d2g_gam_dxCrCr()
dmu_CrNb = Vm * d2g_gam_dxCrNb()
dmu_NbCr = Vm * d2g_gam_dxNbCr()
dmu_NbNb = Vm * d2g_gam_dxNbNb()

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
## TKR5p286

M_Cr = xCr * M_Cr_Cr + xNb * M_Cr_Nb + xNi * M_Cr_Ni + xCr * xNi * M_Cr_CrNi
M_Nb = xCr * M_Nb_Cr + xNb * M_Nb_Nb + xNi * M_Nb_Ni
M_Ni = xCr * M_Ni_Cr + xNb * M_Ni_Nb + xNi * M_Ni_Ni + xCr * xNi * M_Ni_CrNi

print("M = ", np.array([M_Cr, M_Nb, M_Ni]))

# === Chemical Mobilities in FCC Ni ===
## TKR5p292

Z11 =  M_Cr * (1 - xCr)**2    + M_Nb * xCr**2          + M_Ni * xCr**2
Z12 = -M_Cr * (1 - xCr) * xNb - M_Nb * xCr * (1 - xNb) + M_Ni * xCr * xNb
Z21 = -M_Cr * (1 - xCr) * xNb - M_Nb * xCr * (1 - xNb) + M_Ni * xCr * xNb
Z22 =  M_Cr * xNb**2          + M_Nb * (1 - xNb)**2    + M_Ni * xNb**2

D11 = Z11 * dmu_CrCr + Z12 * dmu_NbCr
D12 = Z11 * dmu_CrNb + Z12 * dmu_NbNb
D21 = Z21 * dmu_CrCr + Z22 * dmu_NbCr
D22 = Z21 * dmu_NbCr + Z22 * dmu_NbNb

D = Vm * np.matrix([[D11, D12],[D21, D22]])

print(D)
print("λ =",np.linalg.eig(D)[0])
