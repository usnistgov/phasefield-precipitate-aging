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

## CALPHAD

GCAL_gam = p625.GCAL_gam
GCAL_del = p625.GCAL_del
GCAL_lav = p625.GCAL_lav

GCAL_gam.argtypes = [c_double, c_double]
GCAL_del.argtypes = [c_double, c_double]
GCAL_lav.argtypes = [c_double, c_double]

GCAL_gam.restype = c_double
GCAL_del.restype = c_double
GCAL_lav.restype = c_double

## paraboloid

g_gam = p625.g_gam
g_del = p625.g_del
g_lav = p625.g_lav

g_gam.argtypes = [c_double, c_double]
g_del.argtypes = [c_double, c_double]
g_lav.argtypes = [c_double, c_double]

g_gam.restype = c_double
g_del.restype = c_double
g_lav.restype = c_double

# First Derivatives

## Gamma

dGCAL_gam_dxCr = p625.dGCAL_gam_dxCr
dGCAL_gam_dxNb = p625.dGCAL_gam_dxNb

dGCAL_gam_dxCr.argtypes = [c_double, c_double]
dGCAL_gam_dxNb.argtypes = [c_double, c_double]
dGCAL_gam_dxCr.restype = c_double
dGCAL_gam_dxNb.restype = c_double

dg_gam_dxCr = p625.dg_gam_dxCr
dg_gam_dxNb = p625.dg_gam_dxNb

dg_gam_dxCr.argtypes = [c_double, c_double]
dg_gam_dxNb.argtypes = [c_double, c_double]
dg_gam_dxCr.restype = c_double
dg_gam_dxNb.restype = c_double

## Delta

dGCAL_del_dxCr = p625.dGCAL_del_dxCr
dGCAL_del_dxNb = p625.dGCAL_del_dxNb

dGCAL_del_dxCr.argtypes = [c_double, c_double]
dGCAL_del_dxNb.argtypes = [c_double, c_double]
dGCAL_del_dxCr.restype = c_double
dGCAL_del_dxNb.restype = c_double

dg_del_dxCr = p625.dg_del_dxCr
dg_del_dxNb = p625.dg_del_dxNb

dg_del_dxCr.argtypes = [c_double, c_double]
dg_del_dxNb.argtypes = [c_double, c_double]
dg_del_dxCr.restype = c_double
dg_del_dxNb.restype = c_double

## Laves

dGCAL_lav_dxCr = p625.dGCAL_lav_dxCr
dGCAL_lav_dxNb = p625.dGCAL_lav_dxNb

dGCAL_lav_dxCr.argtypes = [c_double, c_double]
dGCAL_lav_dxNb.argtypes = [c_double, c_double]
dGCAL_lav_dxCr.restype = c_double
dGCAL_lav_dxNb.restype = c_double

dg_lav_dxCr = p625.dg_lav_dxCr
dg_lav_dxNb = p625.dg_lav_dxNb

dg_lav_dxCr.argtypes = [c_double, c_double]
dg_lav_dxNb.argtypes = [c_double, c_double]
dg_lav_dxCr.restype = c_double
dg_lav_dxNb.restype = c_double

# Second Derivatives

## Gamma

d2GCAL_gam_dxCrCr = p625.d2GCAL_gam_dxCrCr
d2GCAL_gam_dxCrNb = p625.d2GCAL_gam_dxCrNb
d2GCAL_gam_dxNbCr = p625.d2GCAL_gam_dxNbCr
d2GCAL_gam_dxNbNb = p625.d2GCAL_gam_dxNbNb

d2GCAL_gam_dxCrCr.restype = c_double
d2GCAL_gam_dxCrNb.restype = c_double
d2GCAL_gam_dxNbCr.restype = c_double
d2GCAL_gam_dxNbNb.restype = c_double

d2g_gam_dxCrCr = p625.d2g_gam_dxCrCr
d2g_gam_dxCrNb = p625.d2g_gam_dxCrNb
d2g_gam_dxNbCr = p625.d2g_gam_dxNbCr
d2g_gam_dxNbNb = p625.d2g_gam_dxNbNb

d2g_gam_dxCrCr.restype = c_double
d2g_gam_dxCrNb.restype = c_double
d2g_gam_dxNbCr.restype = c_double
d2g_gam_dxNbNb.restype = c_double

## Delta

d2GCAL_del_dxCrCr = p625.d2GCAL_del_dxCrCr
d2GCAL_del_dxCrNb = p625.d2GCAL_del_dxCrNb
d2GCAL_del_dxNbCr = p625.d2GCAL_del_dxNbCr
d2GCAL_del_dxNbNb = p625.d2GCAL_del_dxNbNb

d2GCAL_del_dxCrCr.restype = c_double
d2GCAL_del_dxCrNb.restype = c_double
d2GCAL_del_dxNbCr.restype = c_double
d2GCAL_del_dxNbNb.restype = c_double

d2g_del_dxCrCr = p625.d2g_del_dxCrCr
d2g_del_dxCrNb = p625.d2g_del_dxCrNb
d2g_del_dxNbCr = p625.d2g_del_dxNbCr
d2g_del_dxNbNb = p625.d2g_del_dxNbNb

d2g_del_dxCrCr.restype = c_double
d2g_del_dxCrNb.restype = c_double
d2g_del_dxNbCr.restype = c_double
d2g_del_dxNbNb.restype = c_double

## Laves

d2GCAL_lav_dxCrCr = p625.d2GCAL_lav_dxCrCr
d2GCAL_lav_dxCrNb = p625.d2GCAL_lav_dxCrNb
d2GCAL_lav_dxNbCr = p625.d2GCAL_lav_dxNbCr
d2GCAL_lav_dxNbNb = p625.d2GCAL_lav_dxNbNb

d2GCAL_lav_dxCrCr.restype = c_double
d2GCAL_lav_dxCrNb.restype = c_double
d2GCAL_lav_dxNbCr.restype = c_double
d2GCAL_lav_dxNbNb.restype = c_double

d2g_lav_dxCrCr = p625.d2g_lav_dxCrCr
d2g_lav_dxCrNb = p625.d2g_lav_dxCrNb
d2g_lav_dxNbCr = p625.d2g_lav_dxNbCr
d2g_lav_dxNbNb = p625.d2g_lav_dxNbNb

d2g_lav_dxCrCr.restype = c_double
d2g_lav_dxCrNb.restype = c_double
d2g_lav_dxNbCr.restype = c_double
d2g_lav_dxNbNb.restype = c_double

## Diffusivity

D_CrCr = p625.D_CrCr
D_CrNb = p625.D_CrNb
D_NbCr = p625.D_NbCr
D_NbNb = p625.D_NbNb

D_CrCr.argtypes = [c_double, c_double]
D_CrNb.argtypes = [c_double, c_double]
D_NbCr.argtypes = [c_double, c_double]
D_NbNb.argtypes = [c_double, c_double]

D_CrCr.restype = c_double
D_CrNb.restype = c_double
D_NbCr.restype = c_double
D_NbNb.restype = c_double

## Gaussian Enrichment

bellCurve = bell.bell_curve
bellCurve.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
bellCurve.restype = c_double
