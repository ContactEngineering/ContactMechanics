#
# Copyright 2017 Lars Pastewka
#           2015 Till Junge
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
objective function and jacobians for hurst estimation
"""
from sympy import Symbol, pprint
import sympy


q = Symbol('q', positive=True)
denom = Symbol('denom', positive=True)
C = Symbol('C', positive=True)
C0 = Symbol('C_0', positive=True)
H = Symbol('H', positive=True)
cor = Symbol('K', positive=True)


obj = (C-C0*q**(-2-2*H))**2/denom
pprint(obj)

jacob = [sympy.diff(obj, var) for var in (H, C0)]
print("jacobian:")
pprint(jacob)

sol_C0 = sympy.solve(jacob[1], C0)
pprint(sol_C0)

fprime_h = sympy.diff(jacob[0], H)

pprint(sympy.simplify(fprime_h))


pprint(sympy.diff(q**(-2-2*H), H))
