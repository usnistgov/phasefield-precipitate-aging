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

# This Python script generates expressions for a "bell curve" zone of chemical enrichment


from sympy import exp, integrate, simplify
from sympy.utilities.codegen import codegen
from sympy.abc import a, b, x, sigma

bell_f = exp(-0.5 * (x / sigma)**2)
bell_F = integrate(bell_f, x)
bell_a = simplify(bell_F.subs(x, b) - bell_F.subs(x, a)) / (b - a)

codegen([('bell_curve', bell_f), ('bell_integral', bell_F), ('bell_average', bell_a)],
        language='C', prefix='enrichment', project='PrecipitateAging', to_files=True)
