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

# This Python script generates expressions for homogeneous classical nucleation
# assuming cylindrical embryos with a height of one FCC unit cell.

from sympy import Eq, Integral, N, init_printing, pprint, symbols
from sympy import diff, exp, integrate, simplify, solve, sqrt
from sympy.utilities.codegen import codegen
from sympy.abc import D, F, L, T, W, a, c, k, r, v, x, gamma, mu, sigma
from sympy.core.numbers import pi
c0 = symbols('c0')
# init_printing()

A = 2 * pi * (r * a + r**2)
V = pi * a * r**2
G = A * gamma - V * F
dG = diff(G, r)
# pprint(Eq(symbols('G'), G))

rc = simplify(solve(Eq(dG, 0), r)[0])
# pprint(Eq(symbols('r*'), rc))

Gc = simplify(G.subs(r, rc))
# pprint(Eq(symbols('G*'), Gc))

Vc = pi * a * rc**2
rho = pi / (3 * sqrt(2))
Nc = (rho * Vc / v)
# pprint(Eq(symbols('N*'), Nc))

Z = simplify(sqrt(Gc / (3 * pi * k * T * Nc**2)))
# pprint(Eq(symbols('Z'), Z))

B = simplify(A.subs(r, rc) * D * c0 / a**4)
# pprint(Eq(symbols('B*'), B))

N = L * W * a * pi / (3 * v * sqrt(2))
k1 = simplify(B * N * Z)
# pprint(Eq(symbols('k1'), k1))

dc = c - c0
k2 = simplify(Gc / (k * T))
# pprint(Eq(symbols('k2'), k2))

# === BELL CURVE ===

from sympy.abc import a, b, x, sigma

bell_f = exp(-0.5 * (x / sigma)**2)
bell_F = integrate(bell_f, x)
bell_a = simplify(bell_F.subs(x, b) - bell_F.subs(x, a)) / (b - a)

codegen([
    # Constants
    ('k1', k1), ('k2', k2),
    # Bell curves
    ('bell_curve', bell_f), ('bell_integral', bell_F), ('bell_average', bell_a)
        ], language='C', prefix='../src/nucleation', project='PrecipitateAging', to_files=True)
