# -*- coding: utf-8 -*-

# This Python script generates expressions for a "bell curve" zone of chemical enrichment


from sympy import exp, integrate, simplify
from sympy.utilities.codegen import codegen
from sympy.abc import a, b, x, sigma

bell_f = exp(-0.5 * (x / sigma)**2)
bell_F = integrate(bell_f, x)
bell_a = simplify(bell_F.subs(x, b) - bell_F.subs(x, a)) / (b - a)

codegen([('bell_curve', bell_f), ('bell_integral', bell_F), ('bell_average', bell_a)],
        language='C', prefix='enrichment', project='PrecipitateAging', to_files=True)
