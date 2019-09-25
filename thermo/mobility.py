# -*- coding: utf-8 -*-

from constants import *
from pyCinterface import *

xCr = matrixMinCr
xNb = matrixMinNb

mCC = M_CrCr(xCr, xNb)
mCN = M_CrNb(xCr, xNb)
mNC = M_NbCr(xCr, xNb)
mNN = M_NbNb(xCr, xNb)

print("xCr = {0:.4g}".format(xCr))
print("xNb = {0:.4g}\n".format(xNb))
print("M_CrCr = {0:.4g}".format(mCC))
print("M_CrNb = {0:.4g}".format(mCN))
print("M_NbCr = {0:.4g}".format(mNC))
print("M_NbNb = {0:.4g}".format(mNN))
print("===================\n")

xCr = matrixMaxCr
xNb = matrixMaxNb

mCC = M_CrCr(xCr, xNb)
mCN = M_CrNb(xCr, xNb)
mNC = M_NbCr(xCr, xNb)
mNN = M_NbNb(xCr, xNb)

print("xCr = {0:.4g}".format(xCr))
print("xNb = {0:.4g}\n".format(xNb))

print("M_CrCr = {0:.4g}".format(mCC))
print("M_CrNb = {0:.4g}".format(mCN))
print("M_NbCr = {0:.4g}".format(mNC))
print("M_NbNb = {0:.4g}".format(mNN))
print("===================\n")

xCr = enrichMinCr
xNb = enrichMinNb

mCC = M_CrCr(xCr, xNb)
mCN = M_CrNb(xCr, xNb)
mNC = M_NbCr(xCr, xNb)
mNN = M_NbNb(xCr, xNb)

print("xCr = {0:.4g}".format(xCr))
print("xNb = {0:.4g}\n".format(xNb))

print("M_CrCr = {0:.4g}".format(mCC))
print("M_CrNb = {0:.4g}".format(mCN))
print("M_NbCr = {0:.4g}".format(mNC))
print("M_NbNb = {0:.4g}".format(mNN))
print("===================\n")

xCr = enrichMaxCr
xNb = enrichMaxNb

mCC = M_CrCr(xCr, xNb)
mCN = M_CrNb(xCr, xNb)
mNC = M_NbCr(xCr, xNb)
mNN = M_NbNb(xCr, xNb)

print("xCr = {0:.4g}".format(xCr))
print("xNb = {0:.4g}\n".format(xNb))

print("M_CrCr = {0:.4g}".format(mCC))
print("M_CrNb = {0:.4g}".format(mCN))
print("M_NbCr = {0:.4g}".format(mNC))
print("M_NbNb = {0:.4g}".format(mNN))

# === Phase Mobility ===

print("")

D11 = mCC * d2g_gam_dxCrCr() + mCN * d2g_gam_dxCrNb()
D22 = mNC * d2g_del_dxNbCr() + mNN * d2g_del_dxNbNb()
Mphi = D11 * Vm / (2.5e-9 * 2.5e-9 * RT)
print("D ≅", D11)
print("RT ≅", RT)
print("M_phi ≅", Mphi)
