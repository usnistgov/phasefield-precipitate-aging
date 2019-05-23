# -*- coding: utf-8 -*-

# This Python script generates expressions for a "bell curve" zone of chemical enrichment

from sympy import exp, integrate, simplify, init_printing, pprint, symbols
from sympy import solve, Eq, N
from sympy.utilities.codegen import codegen
from sympy.abc import A, a, b, x
sigma, xe, x0 = symbols('sigma x_e x_0', positive=True)
init_printing()

bell_f = A * (exp(-0.5 * (x / sigma)**2) - 1) + xe
bell_F = integrate(bell_f, x)
bell_a = simplify(bell_F.subs(x, b) - bell_F.subs(x, a)) / (b - a)
coeff = simplify(solve(Eq(x0, bell_a), A))[0]
bell_f = simplify(bell_f.subs(A, coeff))

codegen(('bell_curve', bell_f), language='C', prefix='enrichment', project='PrecipitateAging', to_files=True)

"""
# === Check Enrichment ===
import numpy as np
import matplotlib.pyplot as plt

if True:
    xlo, xhi = (-0.5e-6, 0.5e-6)
    c0, cE = (0.025, 0.168)
    w = 150e-9
    # pprint(bell.subs([(x0, c0), (xe, cE), (a, xlo), (b, xhi), (sigma, w)]))

    pos = np.linspace(xlo, xhi, 256)
    bell = np.empty_like(pos)
    for i in range(len(pos)):
        ans = bell_f.subs([(x0, c0), (xe, cE), (a, xlo), (b, xhi), (sigma, w), (x, pos[i])])
        bell[i] = ans

    print("Mean value is {0}".format(np.mean(bell)))

    plt.plot(pos, bell)
    plt.plot((pos[0], pos[-1]), (c0, c0))
    plt.plot((pos[0], pos[-1]), (cE, cE))
    plt.savefig("enrichment.png", dpi=400, bbox_inches="tight")
    plt.close()
"""
