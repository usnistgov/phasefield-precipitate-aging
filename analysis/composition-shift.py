# -*- coding: utf-8 -*-

# Usage: python math/composition-shift.py

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CALPHAD_energies import *

from sympy import Eq, init_printing, Matrix, pprint, solve_linear_system, symbols

init_printing()
P_delta = 2 * s_delta / r_delta
P_laves = 2 * s_laves / r_laves

pprint(Eq(symbols("x_alpha_Cr"), xe_gam_Cr + DXAB(P_delta, P_laves)))
pprint(Eq(symbols("x_alpha_Nb"), xe_gam_Nb + DXAC(P_delta, P_laves)))
pprint(Eq(symbols("x_beta_Cr"), xe_del_Cr + DXBB(P_delta, P_laves)))
pprint(Eq(symbols("x_beta_Nb"), xe_del_Nb + DXBC(P_delta, P_laves)))
pprint(Eq(symbols("x_gamma_Cr"), xe_lav_Cr + DXGB(P_delta, P_laves)))
pprint(Eq(symbols("x_gamma_Nb"), xe_lav_Nb + DXGC(P_delta, P_laves)))

pprint(Eq(symbols("x_Cr"), levers[y]))
pprint(Eq(symbols("x_Nb"), levers[x]))
