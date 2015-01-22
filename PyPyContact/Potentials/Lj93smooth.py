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

import Lj93


class Lj93smooth(Lj93.LJ93):
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
    def __init__(self, epsilon, sigma, gamma=None, r_t=None):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε
        sigma   -- Lennard-Jones distance parameter σ
        gamma   -- (default None) Work of adhesion, defaults to epsilon
        r_t     -- (default None) transition point, defaults to r_min
        """
        self.eps = epsilon
        self.sig = sigma
        self.gamma = gamma if gamma is not None else epsilon
        self.r_t = r_t if r_t is not None else self.r_min


    def eval_poly_and_cutoff(self):
        """ Computes the coefficients of the spline and the cutoff based on σ
            and ε. Since this is a non-linear system of equations, this requires
            some numerics
            The equations derive from the continuity conditions in the class's
            docstring. A few simplifications can be made to conditions (1), (2),
            and (6):

            (1): C_1 = -V_l' (r_t)

            (2): C_2 = -V_l''(r_t)

            (6): C_0 = - γ + C_1*Δr_m + C_2/2*Δr_m^2
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
        """
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sympy import Symbol, pprint
    import sympy
    C0 = Symbol('C0')
    C1 = Symbol('C1')
    C2 = Symbol('C2')
    C3 = Symbol('C3')
    C4 = Symbol('C4')
    dr_c = Symbol('Δrc')
    dr1, dr2 = sympy.solve(    - C2      - 2*C3*dr_c  - 3*C4*dr_c**2, dr_c)
    pprint(dr1)
    pprint(dr2)
    pprint(sympy.solve((-C1 - C2*dr_c -   C3*dr_c**2 - C4*dr_c**3).subs(dr_c, dr1), dr1))
