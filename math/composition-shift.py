#!/usr/bin/python
# -*- coding: utf-8 -*-

#####################################################################################
# This software was developed at the National Institute of Standards and Technology #
# by employees of the Federal Government in the course of their official duties.    #
# Pursuant to title 17 section 105 of the United States Code this software is not   #
# subject to copyright protection and is in the public domain. NIST assumes no      #
# responsibility whatsoever for the use of this code by other parties, and makes no #
# guarantees, expressed or implied, about its quality, reliability, or any other    #
# characteristic. We would appreciate acknowledgement if the software is used.      #
#                                                                                   #
# This software can be redistributed and/or modified freely provided that any       #
# derivative works bear some notice that they are derived from it, and any modified #
# versions bear some notice that they have been modified.                           #
#####################################################################################

# Usage: python math/composition-shift.py

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from CALPHAD_energies import *

from sympy import Eq, init_printing, pprint
init_printing()
P_beta  = 2 * s_delta / r_delta
P_gamma = 2 * s_laves / r_laves

pprint(Eq(symbols('x_alpha_Cr'), xe_gam_Cr + DXAB(s_delta, r_delta, s_laves, r_laves)))
pprint(Eq(symbols('x_alpha_Nb'), xe_gam_Nb + DXAC(s_delta, r_delta, s_laves, r_laves)))
pprint(Eq(symbols('x_beta_Cr'),  xe_del_Cr + DXBB(s_delta, r_delta, s_laves, r_laves)))
pprint(Eq(symbols('x_beta_Nb'),  xe_del_Nb + DXBC(s_delta, r_delta, s_laves, r_laves)))
pprint(Eq(symbols('x_gamma_Cr'), xe_lav_Cr + DXGB(s_delta, r_delta, s_laves, r_laves)))
pprint(Eq(symbols('x_gamma_Nb'), xe_lav_Nb + DXGC(s_delta, r_delta, s_laves, r_laves)))
