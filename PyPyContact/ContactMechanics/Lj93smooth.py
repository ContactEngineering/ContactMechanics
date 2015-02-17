#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   Lj93smooth.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   21 Jan 2015
#
# @brief  9-3 Lennard-Jones potential with forces splined to zero from
# the minimum of the potential using a cubic spline.
#
# @section LICENCE
#
#  Copyright (C) 2015 Till Junge
#
# PyPyContact is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# PyPyContact is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

import scipy.optimize
import numpy as np

from . import Lj93

class LJ93smooth(Lj93.LJ93):
    """ 9-3 Lennard-Jones potential with forces splined to zero from
        the minimum of the potential using a fourth order spline. The 9-3
        Lennard-Jones interaction potential is often used to model the interac-
        tion between a continuous solid wall and the atoms/molecules of a li-
        quid.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        decoupled 9-3:
        V_lγ (r) = V_l(r) - V_l(r_t) + γ

        spline: (Δr = r-r_t)
        V_s(Δr) = C0 - C1*Δr
                  - 1/2*C2*Δr**2
                  - 1/3*C3*Δr**3
                  - 1/4*C4*Δr**4

        The formulation allows to choose the contact stiffness (the original
        repulsive LJ zone) independently from the work of adhesion (the energy
        well). By default, the work of adhesion equals epsilon and the transi-
        tion point r_t between LJ and spline is at the minimumm, however they
        can be chosen freely.

        The spline is chosen to guarantee continuity of the second derivative of
        the potential, leading to the following conditions:
        (1): V_s' (0)    =  V_lγ' (r_t)
        (2): V_s''(0)    =  V_lγ''(r_t)
        (3): V_s  (Δr_c) =  0, where Δr_c = r_c-r_t
        (4): V_s' (Δr_c) =  0
        (5): V_s''(Δr_c) =  0
        (6): V_s  (Δr_m) = -γ, where Δr_m = r_min-r_t
        The unknowns are C_i (i in {0,..4}) and r_t
    """
    name = 'lj9-3smooth'
    def __init__(self, epsilon, sigma, gamma=None, r_t=None, softfail=False):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε (careful, not work of adhesion
                   in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        gamma   -- (default ε) Work of adhesion, defaults to ε
        r_t     -- (default r_min) transition point, defaults to r_min
        """
        self.eps = epsilon
        self.sig = sigma
        self.gamma = gamma if gamma is not None else -self.naive_min
        self.r_t = r_t if r_t is not None else self.r_min
        self.r_c = None
        if self.gamma*self.eps < 0:
            raise self.PotentialError(
                ("ε and γ should have the same sign."
                 "you specified ε = {} and γ = {}. Is your γ "
                 "positive (as it should be)?").format(
                     self.eps, self.gamma))
        ## coefficients of the spline
        self.coeffs = np.zeros(5)
        self.eval_poly_and_cutoff(softfail=softfail)

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', ε = {0.eps}, σ = "
                "{0.sig}{1}{2}").format(
                    self,
                    ", γ = {.gamma}".format(self) if has_gamma else "",
                    ", r_t = {}".format(
                        self.r_t if has_r_t else "r_min"))

    def evaluate(self, r, pot=True, forces=False, curb=False):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        r = np.array(r)
        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        V   = np.zeros_like(r) if pot    else self.SliceableNone()
        dV  = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb   else self.SliceableNone()

        sl_inner = r < self.r_t
        ## little hack to work around numpy bug
        if np.array_equal(sl_inner, np.array([True])):
            V, dV, ddV = self.naive_V(r, pot, forces, curb)
        else:
            V[sl_inner], dV[sl_inner], ddV[sl_inner] = self.naive_V(
                r[sl_inner], pot, forces, curb)
        V[sl_inner] -= self.offset

        sl_outer = r < self.r_c * np.logical_xor(True, sl_inner)
        ## little hack to work around numpy bug
        if np.array_equal(sl_outer, np.array([True])):
            V, dV, ddV = self.spline_V(r, pot, forces, curb)
        else:
            V[sl_outer], dV[sl_outer], ddV[sl_outer] = self.spline_V(
                r[sl_outer], pot, forces, curb)

        return (V    if pot    else None,
                dV   if forces else None,
                ddV  if curb   else None)

    def spline_V(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the spline part and its derivatives of the potential.
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        V = dV = ddV = None
        dr = r-self.r_t
        if pot:
            V = self.poly(dr)
        if forces:
            dV = -self.dpoly(dr) ## Forces are the negative gradient
        if curb:
            ddV = self.ddpoly(dr)
        return (V, dV, ddV)

    def eval_poly_and_cutoff(self, xtol=1e-10, softfail=False):
        """ Computes the coefficients of the spline and the cutoff based on σ
            and ε. Since this is a non-linear system of equations, this requires
            some numerics
            The equations derive from the continuity conditions in the class's
            docstring. A few simplifications can be made to conditions (1), (2),
            and (6):

            (1): C_1 = -V_l' (r_t)

            (2): C_2 = -V_l''(r_t)

            (6): C_0 = + γ + C_1*Δr_m + C_2/2*Δr_m^2
                       + C_3/3*Δr_m^3 + C_4/4*Δr_m^4
            Also, we have some info about the shape of the spline:
            since Δr_c is both an inflection point and an extremum and it has to
            be on the right of the minimum, the minimum has to be the leftmost
            extremum of the spline and therefore the global minimum. It follows:
            1) C_4 < 0
            2) we know sgn(C_1) and sgn(C_2)
            3) any reasonable choice of r_t leads to a Δr_c > 0

            from (5), we get the 2 possible solutions for Δr_c(C_2, C_3, C_4):
                   ⎡         ________________   ⎛        ________________⎞ ⎤
                   ⎢        ╱              2    ⎜       ╱              2 ⎟ ⎥
            Δr_c = ⎢-C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃    -⎝C₃ + ╲╱  -3⋅C₂⋅C₄ + C₃  ⎠ ⎥
                   ⎢─────────────────────────, ────────────────────────────⎥
                   ⎣           3⋅C₄                        3⋅C₄            ⎦
            This however seems to lead to intractable polynomial equations
            (minutes of sympy without finding a solution, ran out of patience)

            Numerical approach: equations (1) and (2) are evauated immediately,
            Then the system of equations

                   ⎡                                   2         ⎤
                   ⎢          -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc          ⎥
                   ⎢                                             ⎥
                   ⎢                            2         3      ⎥
                   ⎢       -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc       ⎥
                   ⎢                                             ⎥
                   ⎢                       2         3         4 ⎥
            F(x) = ⎢                 C₂⋅Δrc    C₃⋅Δrc    C₄⋅Δrc  ⎥ = 0
                   ⎢   C₀ - C₁⋅Δrc - ─────── - ─────── - ─────── ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢                                             ⎥
                   ⎢                    2         3         4    ⎥
                   ⎢              C₂⋅Δrm    C₃⋅Δrm    C₄⋅Δrm     ⎥
                   ⎢C₀ - C₁⋅Δrm - ─────── - ─────── - ─────── - γ⎥
                   ⎣                 2         3         4       ⎦
            is solved. Note that a positive work of adhesion (sticky surface)
            has a negative value for γ. The jacobian is:
                   ⎡                 2                                  ⎤
                   ⎢0  -2⋅Δrc  -3⋅Δrc           -2⋅C₃ - 6⋅C₄⋅Δrc        ⎥
                   ⎢                                                    ⎥
                   ⎢       2        3                               2   ⎥
                   ⎢0  -Δrc     -Δrc       -C₂ - 2⋅C₃⋅Δrc - 3⋅C₄⋅Δrc    ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
            G(x) = ⎢   -Δrc    -Δrc                          2         3⎥
                   ⎢1  ──────  ──────   -C₁ - C₂⋅Δrc - C₃⋅Δrc  - C₄⋅Δrc ⎥
                   ⎢     3       4                                      ⎥
                   ⎢                                                    ⎥
                   ⎢       3       4                                    ⎥
                   ⎢   -Δrm    -Δrm                                     ⎥
                   ⎢1  ──────  ──────                  0                ⎥
                   ⎣     3       4                                      ⎦

            with x = [C₀  C₃  C₄  Δrc]

        Keyword Arguments:
        xtol -- tolerance for numerical solution. Is multiplied by ε internally.

        """
        # known coeffs
        trash, dV, ddV = self.naive_V(self.r_t, pot=False, forces=True, curb=True)
        C1 = self.coeffs[1] = -dV
        C2 = self.coeffs[2] = -ddV
        gam = -self.gamma
        r_t = self.r_t
        dr_m = self.r_min - r_t

        def obj_fun(x):
            C0, C3, C4, dr_c = x
            return np.array(
                [            - C2           - 2*C3 * dr_c  - 3*C4 * dr_c**2,
                    -C1      - C2 * dr_c    -   C3 * dr_c**2 - C4 * dr_c**3,
                 C0 -C1*dr_c - C2/2*dr_c**2 -   C3/3*dr_c**3 - C4/4*dr_c**4,
                 C0 -C1*dr_m - C2/2*dr_m**2 -   C3/3*dr_m**3 - C4/4*dr_m**4
                  - gam])
        def jacobian(x):
            C0, C3, C4, dr_c = x
            return np.array(
             [[0,    -2*dr_c, -3*dr_c**2,            -2*C3       -6*C4*dr_c   ],
              [0,   -dr_c**2,   -dr_c**3,        -C2 -2*C3*dr_c  -3*C4*dr_c**2],
              [1, -dr_c**3/3, -dr_c**4/4, -C1 -C2*dr_c -C3*dr_c**2 -C4*dr_c**3],
              [1, -dr_m**3/3, -dr_m**4/4,                                   0]])

        C3guess = -C2
        C4guess = self.eps/2

        x0 = np.array([-gam, C3guess, 0, max(2.5*self.sig, 2*self.r_min)-r_t])
        options = dict(xtol=self.eps*xtol)
        sol = scipy.optimize.root(obj_fun, x0, jac=jacobian, options=options)
        if True or sol.success:
            self.coeffs[0], self.coeffs[3], self.coeffs[4], self.r_c = sol.x
            self.r_c += self.r_t
            ## !!WARNING!! poly is 'backwards': poly = [C4, C3, C2, C1, C0] and
            ## all coeffs except C0 have the wrong sign, furthermore they are
            ## divided by their order
            polycoeffs = [-self.coeffs[4]/4,
                          -self.coeffs[3]/3,
                          -self.coeffs[2]/2,
                          -self.coeffs[1],
                           self.coeffs[0]]
            self.poly   = np.poly1d(polycoeffs)
            self.dpoly  = np.polyder(self.poly)
            self.ddpoly = np.polyder(self.dpoly)
            self.offset = -(self.spline_V(self.r_t)[0] - self.naive_V(self.r_t)[0])
        else:
            err_str = ("Evaluation of spline for potential '{}' failed. Please check"
                 " whether the inputs make sense").format(self)
            if softfail:
                print("WARNING! {}".format(err_str))
            else:
                raise self.PotentialError(err_str)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sympy import Symbol, pprint, solve_poly_system, Matrix, zeros
    import sympy
    C0 = Symbol('C0')
    C1 = Symbol('C1')
    C2 = Symbol('C2')
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    dr_c = Symbol('Δrc')
    dr_m = Symbol('Δrm')
    gam = Symbol('γ')
    dr1, dr2 = sympy.solve(    - C2      - 2*C3*dr_c  - 3*C4*dr_c**2, dr_c)

    eq5 =             - C2           - 2*C3 * dr_c  - 3*C4 * dr_c**2
    eq4 =    -C1      - C2 * dr_c    -   C3 * dr_c**2 - C4 * dr_c**3
    eq3 = C0 -C1*dr_c - C2/2*dr_c**2 -   C3/3*dr_c**3 - C4/4*dr_c**4
    eq6 = C0 -C1*dr_m - C2/2*dr_m**2 -   C3/3*dr_m**3 - C4/4*dr_m**4 + gam

    ## apparently not easily solvable
    #solve_poly_system([eq5, eq4, eq3, eq6], dr_c, C4, C3, C0)
    F = Matrix([[eq5, eq4, eq3, eq6]])
    pprint(F.T)
    vs = Matrix([[C0, C3, C4, dr_c]])
    pprint (vs)
    size = len(vs)
    gradient = zeros(size, size)
    for i in range(size):
        for j in range(size):
            gradient[i,j] = sympy.diff(F[i], vs[j])
    pprint(gradient)
    print(gradient)

    epsilon, sigma, r_t, gamma = 1.2, 4, 4, 1.3
    ## print(LJ93smooth(epsilon, sigma))
    ## print(LJ93smooth(epsilon, sigma).poly)
    ## print(LJ93smooth(epsilon, sigma, gamma=gamma))
    ## print(LJ93smooth(epsilon, sigma, gamma=gamma).poly)
    ## print(LJ93smooth(epsilon, sigma, r_t =r_t))
    ## print(LJ93smooth(epsilon, sigma, gamma=gamma, r_t =r_t))


    import matplotlib.pyplot as plt
    f = plt.figure()
    p_ax = f.add_subplot(131)
    f_ax = f.add_subplot(132)
    c_ax = f.add_subplot(133)

    pot = LJ93smooth(epsilon, sigma, gamma=-epsilon, r_t=r_t)

    x = np.arange(.8*sigma, 2*sigma, .0025*sigma)
    V, dV, ddV = pot.evaluate(x, pot=True, forces=True, curb=True)
    color = p_ax.plot(x, V, label="bla")[0].get_color()
    sig_pts = [pot.r_t, pot.r_min, pot.r_c]
    V_t, dV_t, ddV_t = pot.evaluate(sig_pts, True, True, True)


    p_ax.scatter(sig_pts, V_t, marker='x', c=color)
    f_ax.plot(x, dV, c=color)
    f_ax.scatter(sig_pts, dV_t, marker='x', c=color)
    c_ax.plot(x, ddV, c=color)
    c_ax.scatter(sig_pts, ddV_t, marker='x', c=color)


    p_ax.legend(loc='best')
    x_range = p_ax.get_xlim()
    f_ax.set_xlim(x_range)
    c_ax.set_xlim(x_range)
    py_range = p_ax.get_ylim()
    fy_range = f_ax.get_ylim()
    cy_range = c_ax.get_ylim()

    V, dV, ddV = pot.spline_V(x, pot=True, forces=True, curb=True)
    color = p_ax.plot(x, V, label="bla", ls='--')[0].get_color()
    f_ax.plot(x, dV, c=color, ls='--')
    c_ax.plot(x, ddV, c=color, ls='--')

    V, dV, ddV = pot.naive_V(x, pot=True, forces=True, curb=True)
    color = p_ax.plot(x, V, label="bla", ls='--')[0].get_color()
    f_ax.plot(x, dV, c=color, ls='--')
    c_ax.plot(x, ddV, c=color, ls='--')

    p_ax.set_ylim(py_range)
    f_ax.set_ylim(fy_range)
    c_ax.set_ylim(cy_range)
    p_ax.grid(True)
    f_ax.grid(True)
    c_ax.grid(True)
    plt.show()
