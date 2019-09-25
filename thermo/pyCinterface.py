# -*- coding:utf-8 -*-

from ctypes import CDLL, c_double

bell = CDLL("./enrichment.so")
p625 = CDLL("./parabola625.so")

# Control points

xe1A = p625.xe_gam_Nb
xe2A = p625.xe_gam_Cr
xe1B = p625.xe_del_Nb
xe2B = p625.xe_del_Cr
xe1C = p625.xe_lav_Nb
xe2C = p625.xe_lav_Cr

xe1A.restype = c_double
xe2A.restype = c_double
xe1B.restype = c_double
xe2B.restype = c_double
xe1C.restype = c_double
xe2C.restype = c_double

# Define free energy functions

g_gam = p625.g_gam
g_del = p625.g_del
g_lav = p625.g_lav

g_gam.argtypes = [c_double, c_double]
g_del.argtypes = [c_double, c_double]
g_lav.argtypes = [c_double, c_double]

g_gam.restype = c_double
g_del.restype = c_double
g_lav.restype = c_double

CALPHAD_gam = p625.CALPHAD_gam
CALPHAD_del = p625.CALPHAD_del
CALPHAD_lav = p625.CALPHAD_lav

CALPHAD_gam.argtypes = [c_double, c_double]
CALPHAD_del.argtypes = [c_double, c_double]
CALPHAD_lav.argtypes = [c_double, c_double]

CALPHAD_gam.restype = c_double
CALPHAD_del.restype = c_double
CALPHAD_lav.restype = c_double

# First Derivatives

## Gamma

dg_gam_dxCr = p625.dg_gam_dxCr
dg_gam_dxNb = p625.dg_gam_dxNb

dg_gam_dxCr.argtypes = [c_double, c_double]
dg_gam_dxNb.argtypes = [c_double, c_double]
dg_gam_dxCr.restype = c_double
dg_gam_dxNb.restype = c_double

## Delta

dg_del_dxCr = p625.dg_del_dxCr
dg_del_dxNb = p625.dg_del_dxNb

dg_del_dxCr.argtypes = [c_double, c_double]
dg_del_dxNb.argtypes = [c_double, c_double]
dg_del_dxCr.restype = c_double
dg_del_dxNb.restype = c_double

## Laves

dg_lav_dxCr = p625.dg_lav_dxCr
dg_lav_dxNb = p625.dg_lav_dxNb

dg_lav_dxCr.argtypes = [c_double, c_double]
dg_lav_dxNb.argtypes = [c_double, c_double]
dg_lav_dxCr.restype = c_double
dg_lav_dxNb.restype = c_double

# Second Derivatives

## Gamma

d2g_gam_dxCrCr = p625.d2g_gam_dxCrCr
d2g_gam_dxCrNb = p625.d2g_gam_dxCrNb
d2g_gam_dxNbCr = p625.d2g_gam_dxNbCr
d2g_gam_dxNbNb = p625.d2g_gam_dxNbNb

d2g_gam_dxCrCr.restype = c_double
d2g_gam_dxCrNb.restype = c_double
d2g_gam_dxNbCr.restype = c_double
d2g_gam_dxNbNb.restype = c_double


## Delta

d2g_del_dxCrCr = p625.d2g_del_dxCrCr
d2g_del_dxCrNb = p625.d2g_del_dxCrNb
d2g_del_dxNbCr = p625.d2g_del_dxNbCr
d2g_del_dxNbNb = p625.d2g_del_dxNbNb

d2g_del_dxCrCr.restype = c_double
d2g_del_dxCrNb.restype = c_double
d2g_del_dxNbCr.restype = c_double
d2g_del_dxNbNb.restype = c_double

## Laves

d2g_lav_dxCrCr = p625.d2g_lav_dxCrCr
d2g_lav_dxCrNb = p625.d2g_lav_dxCrNb
d2g_lav_dxNbCr = p625.d2g_lav_dxNbCr
d2g_lav_dxNbNb = p625.d2g_lav_dxNbNb

d2g_lav_dxCrCr.restype = c_double
d2g_lav_dxCrNb.restype = c_double
d2g_lav_dxNbCr.restype = c_double
d2g_lav_dxNbNb.restype = c_double

## Mobility

M_CrCr = p625.M_CrCr
M_CrCr.argtypes = [c_double, c_double]
M_CrCr.restype = c_double
M_CrNb = p625.M_CrNb
M_CrNb.argtypes = [c_double, c_double]
M_CrNb.restype = c_double
M_NbCr = p625.M_NbCr
M_NbCr.argtypes = [c_double, c_double]
M_NbCr.restype = c_double
M_NbNb = p625.M_NbNb
M_NbNb.argtypes = [c_double, c_double]
M_NbNb.restype = c_double

## Gaussian Enrichment

bellCurve = bell.bell_curve
bellCurve.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
bellCurve.restype = c_double
