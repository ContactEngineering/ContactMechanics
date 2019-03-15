#
# Copyright 2017, 2019 Lars Pastewka
#           2015-2016 Till Junge
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
useful symbolic expressions for self-affine rough surfaces
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

