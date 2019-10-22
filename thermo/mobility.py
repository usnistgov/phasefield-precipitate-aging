# -*- coding: utf-8 -*-

from constants import *
from pyCinterface import *

comp = ((0.02, 0.30),
        (0.5 * (matrixMinNb + matrixMaxNb), 0.5 * (matrixMinCr + matrixMaxCr)),
        (0.5 * (enrichMinNb + enrichMaxNb), 0.5 * (enrichMinCr + enrichMaxCr)))

for xNb, xCr in comp:
    print("<-- xCr = {0:6.4g}, xNb = {1:6.4g} -->".format(xCr, xNb))

    # === Mobility ===

    mCr = M_Cr(xCr, xNb)
    mNb = M_Nb(xNb)
    mNi = M_Ni(xCr, xNb)
    print("M_Cr = {0:10.4g}, M_Nb = {1:10.4g}, M_Ni = {2:10.4g}\n".format(mCr, mNb, mNi))

    mCC = M_CrCr(xCr, xNb)
    mCN = M_CrNb(xCr, xNb)
    mNN = M_NbNb(xCr, xNb)

    # === Diffusivity ===

    D11 = Vm * (mCC * d2g_gam_dxCrCr() + mCN * d2g_gam_dxCrNb())
    D12 = Vm * (mCC * d2g_gam_dxNbCr() + mCN * d2g_gam_dxNbNb())
    D21 = Vm * (mCN * d2g_gam_dxCrCr() + mNN * d2g_gam_dxCrNb())
    D22 = Vm * (mCN * d2g_gam_dxNbCr() + mNN * d2g_gam_dxNbNb())
    Mphi = D11 * Vm / (2.5e-9 * 2.5e-9 * RT)
    print("D_gam = {0:10.4g} {1:10.4g}\n        {2:10.4g} {3:10.4g}\n".format(D11, D12, D21, D22))
    D11 = Vm * (mCC * d2g_del_dxCrCr() + mCN * d2g_del_dxCrNb())
    D12 = Vm * (mCC * d2g_del_dxNbCr() + mCN * d2g_del_dxNbNb())
    D21 = Vm * (mCN * d2g_del_dxCrCr() + mNN * d2g_del_dxCrNb())
    D22 = Vm * (mCN * d2g_del_dxNbCr() + mNN * d2g_del_dxNbNb())
    print("D_del = {0:10.4g} {1:10.4g}\n        {2:10.4g} {3:10.4g}\n".format(D11, D12, D21, D22))
    D11 = Vm * (mCC * d2g_lav_dxCrCr() + mCN * d2g_lav_dxCrNb())
    D12 = Vm * (mCC * d2g_lav_dxNbCr() + mCN * d2g_lav_dxNbNb())
    D21 = Vm * (mCN * d2g_lav_dxCrCr() + mNN * d2g_lav_dxCrNb())
    D22 = Vm * (mCN * d2g_lav_dxNbCr() + mNN * d2g_lav_dxNbNb())
    print("D_lav = {0:10.4g} {1:10.4g}\n        {2:10.4g} {3:10.4g}\n".format(D11, D12, D21, D22))
