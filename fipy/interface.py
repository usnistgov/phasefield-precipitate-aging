# -*- coding:utf-8 -*-

from ctypes import CDLL, c_double
import numpy as np

p625 = CDLL("../thermo/parabola625.so")

# Control points

xe_gam_Nb = p625.xe_gam_Nb
xe_gam_Cr = p625.xe_gam_Cr
xe_del_Nb = p625.xe_del_Nb
xe_del_Cr = p625.xe_del_Cr

xe_gam_Nb.restype = c_double
xe_gam_Cr.restype = c_double
xe_del_Nb.restype = c_double
xe_del_Cr.restype = c_double

# Define free energy functions

g_gam = p625.g_gam
g_del = p625.g_del

g_gam.argtypes = [c_double, c_double]
g_del.argtypes = [c_double, c_double]

g_gam.restype = c_double
g_del.restype = c_double

def g_gam_v(xCr, xNb):
    return [g_gam(a, b) for a, b in np.ravel(xCr, xNb)]
def g_del_v(xCr, xNb):
    return [g_del(a, b) for a, b in np.ravel(xCr, xNb)]

# First Derivatives

## Gamma

dg_gam_dxCr = p625.dg_gam_dxCr
dg_gam_dxNb = p625.dg_gam_dxNb

dg_gam_dxCr.argtypes = [c_double, c_double]
dg_gam_dxNb.argtypes = [c_double, c_double]
dg_gam_dxCr.restype = c_double
dg_gam_dxNb.restype = c_double

def dg_gam_dxCr_v(xCr, xNb):
    return np.vectorize(dg_gam_dxCr)
def dg_gam_dxNb_v(xCr, xNb):
    return np.vectorize(dg_gam_dxNb)

## Delta

dg_del_dxCr = p625.dg_del_dxCr
dg_del_dxNb = p625.dg_del_dxNb

dg_del_dxCr.argtypes = [c_double, c_double]
dg_del_dxNb.argtypes = [c_double, c_double]
dg_del_dxCr.restype = c_double
dg_del_dxNb.restype = c_double

def dg_del_dxCr_v(xCr, xNb):
    return np.vectorize(dg_del_dxCr)
def dg_del_dxNb_v(xCr, xNb):
    return np.vectorize(dg_del_dxNb)

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

def D_CrCr_v(xCr, xNb):
    return np.vectorize(D_CrCr)
def D_CrNb_v(xCr, xNb):
    return np.vectorize(D_CrNb)
def D_NbCr_v(xCr, xNb):
    return np.vectorize(D_NbCr)
def D_NbNb_v(xCr, xNb):
    return np.vectorize(D_NbNb)

## Fictitious compositions

x_gam_Cr = p625.fict_gam_Cr
x_gam_Nb = p625.fict_gam_Nb
x_del_Cr = p625.fict_del_Cr
x_del_Nb = p625.fict_del_Nb

x_gam_Cr.argtypes = [c_double, c_double, c_double, c_double, c_double]
x_gam_Nb.argtypes = [c_double, c_double, c_double, c_double, c_double]
x_del_Cr.argtypes = [c_double, c_double, c_double, c_double, c_double]
x_del_Nb.argtypes = [c_double, c_double, c_double, c_double, c_double]

x_gam_Cr.restype = c_double
x_gam_Nb.restype = c_double
x_del_Cr.restype = c_double
x_del_Nb.restype = c_double

def x_gam_Cr_v(xCr, xNb, pD, pG, pL):
    return np.vectorize(x_gam_Cr)

def x_gam_Nb_v(xCr, xNb, pD, pG, pL):
    return np.vectorize(x_gam_Nb)

def x_del_Cr_v(xCr, xNb, pD, pG, pL):
    return np.vectorize(x_del_Cr)

def x_del_Nb_v(xCr, xNb, pD, pG, pL):
    return np.vectorize(x_del_Nb)
