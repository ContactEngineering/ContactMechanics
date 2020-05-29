#
# Copyright 2015-2016 Till Junge
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
Trying to figure out under what conditions we can use the 4th order
spline to smoothen out a vdW potential
"""

from PyCo.Adhesion import LJ93smooth
from sympy import Symbol, pprint
import sympy
from copy import deepcopy

r = Symbol('r', positive=True)
dr = Symbol('Δr', real=True)
dgam = Symbol('(Δγ-γ)', negative=True)
gam = Symbol('γ', positive=True)

c_sr = Symbol('C_sr', positive=True)
hama = Symbol('A', positive=True)

dV_t = Symbol('dV_t', real=True)
ddV_t = Symbol('ddV_t', real=True)


vdw = -hama/(12*r**2*sympy.pi)+c_sr/r**8
dvdw = sympy.diff(vdw, r)
ddvdw = sympy.diff(dvdw, r)


radicant = -12 * dgam * ddV_t + 9 * dV_t**2
print('radicant')
pprint(radicant)

print("radicant for van der Waals")
vdw_radicant = radicant.subs({dgam: vdw, dV_t:dvdw, ddV_t:ddvdw})
pprint(vdw_radicant)

#boundaries of validity
print("Roots of radicant")
sol_radicant = sympy.solve(vdw_radicant, r)
for sol in sol_radicant:
    pprint(sol)
    print('is positive: {}'.format(sol.is_positive))
#pprint(sol_radicant)
