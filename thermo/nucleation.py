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
# assuming cylindrical embryos with a haight of one FCC unit cell.

from sympy import diff, Eq, expand, factor, fraction, init_printing, pprint, simplify, solve, sqrt, symbols
from sympy.abc import D, F, L, T, W, a, c, k, r, v, gamma
from sympy.core.numbers import pi
from sympy.utilities.codegen import codegen
c0 = symbols('c0')
init_printing()

A = 2 * pi * (r * a + r**2)
V = pi * a * r**2
G = A * gamma - V * F
dG = diff(G, r)
pprint(Eq(symbols('G'), G))

rc = simplify(solve(Eq(dG, 0), r)[0])
pprint(Eq(symbols('r*'), rc))

Gc = simplify(G.subs(r, rc))
pprint(Eq(symbols('G*'), Gc))

Vc = pi * a * rc**2
rho = pi / (3 * sqrt(2))
Nc = (rho * Vc / v)
pprint(Eq(symbols('N*'), Nc))

Z = simplify(sqrt(Gc / (3 * pi * k * T * Nc**2)))
pprint(Eq(symbols('Z'), Z))

B = simplify(A.subs(r, rc) * D * c0 / a**4)
pprint(Eq(symbols('B*'), B))

N = L * W * a * pi / (3 * v * sqrt(2))
k1 = simplify(B * N * Z)
pprint(Eq(symbols('k1'), k1))

dc = c - c0
k2 = Gc / (k * T)
pprint(Eq(symbols('k2'), k2))
