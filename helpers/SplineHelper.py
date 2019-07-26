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
symbolic calculations for splines
"""

from PyCo.ContactMechanics import LJ93smooth
from sympy import Symbol, pprint, solve_poly_system, Matrix, zeros
import sympy
from copy import deepcopy

# free variable
dr = Symbol('Δr')

# spline parameters
c0 = Symbol('C0', real=True)
c1 = Symbol('C1', real=True)
c2 = Symbol('C2', real=True)
c3 = Symbol('C3', real=True)
c4 = Symbol('C4', real=True)
dr_c = Symbol('Δrc', real=True)
dr_t = Symbol('Δrt', real=True)
dr_m = Symbol('Δrm', real=True)
dgam = Symbol('(Δγ-γ)', negative=True)

# boundary parameters
gam = Symbol('γ', positive=True)
dV_t = Symbol('dV_t', real=True)
ddV_t = Symbol('ddV_t', real=True)
ddV_m = Symbol('ddV_M', real=True)

fun = c0 - c1*dr - c2/2*dr**2 - c3/3*dr**3 - c4/4*dr**4
dfun = sympy.diff(fun, dr)
ddfun = sympy.diff(dfun, dr)
print("Spline:")
pprint(fun)
print("Slope:")
pprint(dfun)
print("Curvature:")
pprint(ddfun)

# boundary conditions (satisfied if equal to zero)
bnds = dict()
bnds[1] = dfun.subs(dr, dr_t) - dV_t
bnds[2] = ddfun.subs(dr, dr_t) - ddV_t
bnds[3] = fun.subs(dr, dr_c)
bnds[4] = dfun.subs(dr, dr_c)
bnds[5] = ddfun.subs(dr, dr_c)

stored_bnds = deepcopy(bnds)

def pbnd(boundaries):
    print("\n")
    for key, bnd in boundaries.items():
        print("Boundary condition {}:".format(key))
        pprint(bnd)

# assuming the origin for Δr is at the cutoff (Δrc):
bnds[3] = bnds[3].subs(dr_c, 0.)  # everything is zero at r_cut
bnds[4] = bnds[4].subs(dr_c, 0.)  # everything is zero at r_cut
bnds[5] = bnds[5].subs(dr_c, 0.)  # everything is zero at r_cut


print()
print('#####################################')
print("For r_t <= r_min:")

# all at once?
coeff_sol = sympy.solve(bnds.values(), [c0, c1, c2, c3, c4])
print('\nCoefficients')
pprint(coeff_sol)

# solving for Δrt
print('substituted polynomial')
polynomial = fun.subs(coeff_sol)
pprint(polynomial)
# Δrm is a minimum (actually, the global minimum), the one not at zero
dpolynomial = sympy.diff(polynomial, dr)
print("substituted polynomial's derivative:")
pprint(dpolynomial)
sols = sympy.solve(dpolynomial, dr)
for sol in sols:
    if sol != 0:
        sol_drm = sol

print("\nsolution for Δr_m")
pprint(sol_drm)

# γ-condition:
γfun = sympy.simplify(polynomial.subs(dr, sol_drm) + gam)

print('\nγ-condition is not solvable analytically.')
print('objective function:')
pprint(γfun)
dγfun = sympy.simplify(sympy.diff(γfun, dr_t))
print('\nobjective function derivative:')
pprint(dγfun)

# not solvable in sympy, but good initial guess for optimisation can be
# obtained for case where r_min = r_t (the default case)
guess_γfun = γfun.subs({"dV_t": 0, "ddV_t": ddV_m})
guess_sol = sympy.solve(guess_γfun, dr_t)[0]


print('\ninitial guess: note that you need to evaluate the curvature at r_min '
      'for the square root te be guaranteed to be real (i.e, than the '
      'curvature and γ have the same sign')
pprint(guess_sol)


print()
print("for usage in code:")
print("\nCoefficients: ", [coeff_sol[c0], coeff_sol[c1], coeff_sol[c2], coeff_sol[c3], coeff_sol[c4]])
print("\nobjective_fun: ", γfun)
print("\nobjective_derivative: ", dγfun)
print("\ninitial guess for Δr_t: ", guess_sol)
print("\nsol for Δr_m: ", sol_drm)


print()
print('#####################################')
print("For r_t > r_min:")
bnds[6] = fun.subs(dr, dr_t) - dgam

# all at once is a mess, better to split the solution:
coeff_sol = sympy.solve(list(bnds.values())[:-1], [c0, c1, c2, c3, c4])
print('\nCoefficients')
pprint(coeff_sol)
print("γ-Condition:")
pprint(bnds[6])
print("γ-Condition, substituted:")
pprint(bnds[6].subs(coeff_sol))
bnd6_sol = sympy.solve(bnds[6].subs(coeff_sol), dr_t)
print("γ-Condition, solved for Δrt (first one is the correct one):")
pprint(bnd6_sol)
print("∂γ/∂(ddV_t) :")
dΔrt = sympy.diff(bnd6_sol[0], ddV_t)
pprint(dΔrt)
print("∂(Δrt)/∂(ddV_t) at zero :")
dΔrt_0 = sympy.limit(dΔrt, ddV_t, 0)
pprint(dΔrt_0)
print("∂(Δrt)/∂(ddV_t) at zero with substitution:")
a = Symbol('a', positive = True)
b = Symbol('b', positive = True)
subs = {dV_t: a/3, dgam:b/-12}
resubs = {a:3*dV_t, b:-12*dgam}
sub_ddrt = dΔrt.subs(subs)
subdΔrt_0 = sympy.limit(sub_ddrt, ddV_t, 0)
pprint(subdΔrt_0)
pprint(subdΔrt_0.subs(resubs))
# solving for Δrt
print('substituted polynomial')
polynomial = sympy.simplify(fun.subs(coeff_sol))
pprint(polynomial)

print()
print('#####################################')
print("For r_t = r_infl:")
bnds[2] = ddfun.subs(dr, dr_t)
bnds[6] = fun.subs(dr, dr_t) - dgam

# all at once is a mess, better to split the solution:
coeff_sol = sympy.solve(list(bnds.values())[:-1], [c0, c1, c2, c3, c4])
print('\nCoefficients')
pprint(coeff_sol)
print("γ-Condition:")
pprint(bnds[6])
print("γ-Condition, substituted:")
pprint(bnds[6].subs(coeff_sol))
bnd6_sol = sympy.solve(bnds[6].subs(coeff_sol), dr_t)
print("γ-Condition, solved for Δrt:")
pprint(bnd6_sol)
# solving for Δrt
print('substituted polynomial')
polynomial = sympy.simplify(fun.subs(coeff_sol))
pprint(polynomial)

