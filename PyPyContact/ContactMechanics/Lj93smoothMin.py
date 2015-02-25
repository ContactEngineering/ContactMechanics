#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Lj93smoothMin.py

@author Till Junge <till.junge@kit.edu>

@date   24 Feb 2015

@brief  Implements a modifies smooth LJ93 potential without singularities
        at the origin. Suitable for ill-behaved minimisations

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import numpy as np

from .Lj93smooth import LJ93smooth


class LJ93smoothMin(LJ93smooth):
    """
    When starting from a bad guess, or with a bad optimizer, sometimes
    optimisations that include potentials with a singularity at the origin
    fail, because the optimizer chooses a bad step direction and length and
    falls into non-physical territory. This class tries to remedy this by
    replacing the singular repulsive part around zero by a linear function.
    """
    name = 'lj9-3smooth-min'

    def __init__(self, epsilon, sigma, gamma=None, r_ti=None, r_t_ls=None):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε (careful, not work of
                   adhesion in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        gamma   -- (default ε) Work of adhesion, defaults to ε
        r_ti    -- (default r_min) transition point between linear function and
                   lj, defaults to r_min
        r_t_ls  -- (default r_min) transition point between lj and spline,
                    defaults to r_min
        """
        super().__init__(epsilon, sigma, gamma, r_t_ls)
        self.r_ti = r_ti if r_ti is not None else self.r_min/2
        self.lin_part = self.compute_linear_part()

    def compute_linear_part(self):
        " evaluates the two coefficients of the linear part of the potential"
        f_val, f_prime, dummy = super().evaluate(self.r_ti, True, True)
        return np.poly1d((float(-f_prime), f_val + f_prime*self.r_ti))

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', ε = {0.eps}, σ = "
                "{0.sig}{1}{2}, r_ti = {0.r_ti}").format(
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
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        r = np.array(r)
        nb_dim = len(r.shape)
        if nb_dim == 0:
            r.shape = (1,)
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        sl_core = r < self.r_ti
        sl_rest = np.logical_not(sl_core)
        # little hack to work around numpy bug
        if np.array_equal(sl_core, np.array([True])):
            return self.lin_pot(r, pot, forces, curb)
        else:
            V[sl_core], dV[sl_core], ddV[sl_core] = self.lin_pot(
                r[sl_core], pot, forces, curb)

        sl_inner = np.logical_and(r < self.r_t, sl_rest)
        sl_rest *= np.logical_not(sl_inner)
        # little hack to work around numpy bug
        if np.array_equal(sl_inner, np.array([True])):
            V, dV, ddV = self.naive_pot(r, pot, forces, curb)
            V -= self.offset
            return V, dV, ddV
        else:
            V[sl_inner], dV[sl_inner], ddV[sl_inner] = self.naive_pot(
                r[sl_inner], pot, forces, curb)
        V[sl_inner] -= self.offset

        sl_outer = np.logical_and(r < self.r_c, sl_rest)
        # little hack to work around numpy bug
        if np.array_equal(sl_outer, np.array([True])):
            V, dV, ddV = self.spline_pot(r, pot, forces, curb)
        else:
            V[sl_outer], dV[sl_outer], ddV[sl_outer] = self.spline_pot(
                r[sl_outer], pot, forces, curb)

        return (V if pot else None,
                dV if forces else None,
                ddV if curb else None)

    def lin_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the linear part and its derivatives of the potential.
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        V = None if pot is False else self.lin_part(r)
        dV = None if forces is False else self.lin_part[1]
        ddV = None if curb is False else 0.
        return V, dV, ddV
