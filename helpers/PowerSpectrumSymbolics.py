#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   PowerSpectrumSymbolics.py

@author Till Junge <till.junge@kit.edu>

@date   01 Jul 2015

@brief  useful symbolic expressions for self-affine rough surfaces

@section LICENCE

 Copyright (C) 2015 Till Junge

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from sympy import Symbol, pprint, solve_poly_system, Matrix, zeros
import sympy


q = Symbol('q', positive=True)
q0 = Symbol('q0', positive=True)
q1 = Symbol('q1', positive=True)
h = Symbol('H', positive=True)
alpha = Symbol('α', positive=True)
beta = Symbol('β', positive=True)
pi = Symbol('π', positive=True)
h_rms = Symbol('h_rms', positive=True)

C = beta**2*q**(-2*(h+1))
print("C(q):")
pprint(C)
print()
expr = 1/(2*pi)*q*C
print("integrand")
pprint(expr)
print()

print('q*C')
pprint(sympy.simplify(C*q))
print()
h2 = sympy.simplify(sympy.integrate(expr, q))#(q, q0, q1)))
print("(h_rms)**2:")
pprint(h2)
print()

h2 = 2*pi*sympy.integrate(q*C, (q, q0, q1))
print("(h_rms)**2:")
pprint(h2)
print()

#beta_sol = sympy.solve(h2-h_rms**2, beta)[0]
#print("β:")
#pprint(beta_sol)
#print()

beta_lim = h2.subs({q1: sympy.oo})
beta_lim_sol = sympy.solve(beta_lim-h_rms**2, beta)[0]
print("for large {}, β:".format(q1))
pprint(beta_lim_sol)
print()

