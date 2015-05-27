#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Lj93.py

@author Till Junge <till.junge@kit.edu>

@date   22 Jan 2015

@brief  9-3 Lennard-Jones potential for wall interactions

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

from . import Potential, SmoothPotential, MinimisationPotential
from . import SimpleSmoothPotential


class LJ93(Potential):
    """ 9-3 Lennard-Jones potential with optional cutoff radius.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        When used with a cutoff radius, the potential is shifted in orden to
        guarantee continuity of forces

        V_lc (r) = V_l (r) - V_l (r_c)
    """

    name = "lj9-3"

    def __init__(self, epsilon, sigma, r_cut=float('inf')):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε
        sigma   -- Lennard-Jones distance parameter σ
        r_cut   -- (default i) optional cutoff radius
        """
        self.eps = epsilon
        self.sig = sigma
        Potential.__init__(self, r_cut)

    def __repr__(self, ):
        return ("Potential '{0.name}': ε = {0.eps}, σ = {0.sig}, "
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else '∞')

    @property
    def r_min(self):
        """convenience function returning the location of the energy minimum

                6 ___  5/6
        r_min = ╲╱ 2 ⋅5   ⋅σ
                ────────────
                     5
        """
        return self.sig*(2*5**5)**(1./6)/5.


    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)

        r_infl = σ
        """
        return self.sig

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated LJ terms for efficiency
                         ⎛   3       9⎞
                         ⎜  σ     2⋅σ ⎟
            V_l(r) =   ε⋅⎜- ── + ─────⎟
                         ⎜   3       9⎟
                         ⎝  r    15⋅r ⎠

                         ⎛   3       9⎞
                         ⎜3⋅σ     6⋅σ ⎟
            V_l'(r) =  ε⋅⎜──── - ─────⎟
                         ⎜  4       10⎟
                         ⎝ r     5⋅r  ⎠

                         ⎛      3       9⎞
                         ⎜  12⋅σ    12⋅σ ⎟
            V_l''(r) = ε⋅⎜- ───── + ─────⎟
                         ⎜     5      11 ⎟
                         ⎝    r      r   ⎠

            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        V = dV = ddV = None
        sig_r3 = (self.sig/r)**3
        sig_r9 = sig_r3**3
        if pot:
            V = self.eps*(2./15*sig_r9 - sig_r3)
        if forces or curb:
            eps_r = self.eps/r
        if forces:
            # Forces are the negative gradient
            dV = eps_r*(6./5*sig_r9 - 3*sig_r3)
        if curb:
            ddV = 12*eps_r/r*(sig_r9 - sig_r3)
        return (V, dV, ddV)


class LJ93smooth(LJ93, SmoothPotential):
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

        The spline is chosen to guarantee continuity of the second derivative
        of the potential, leading to the following conditions:
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
        epsilon -- Lennard-Jones potential well ε (careful, not work of
                   adhesion in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        gamma   -- (default ε) Work of adhesion, defaults to ε
        r_t     -- (default r_min) transition point, defaults to r_min
        """
        LJ93.__init__(self, epsilon, sigma, None)
        SmoothPotential.__init__(self, gamma, r_t)

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', ε = {0.eps}, σ = "
                "{0.sig}{1}{2}").format(
                    self,
                    ", γ = {.gamma}".format(self) if has_gamma else "",
                    ", r_t = {}".format(
                        self.r_t if has_r_t else "r_min"))


class LJ93smoothMin(LJ93smooth, MinimisationPotential):
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
        r_ti    -- (default r_min/2) transition point between linear function
                   and lj, defaults to r_min
        r_t_ls  -- (default r_min) transition point between lj and spline,
                    defaults to r_min
        """
        LJ93smooth.__init__(self, epsilon, sigma, gamma, r_t_ls)
        MinimisationPotential.__init__(self, r_ti)

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', ε = {0.eps}, σ = "
                "{0.sig}{1}{2}, r_ti = {0.r_ti}").format(
                    self,
                    ", γ = {.gamma}".format(self) if has_gamma else "",
                    ", r_t = {}".format(
                        self.r_t if has_r_t else "r_min"))


class LJ93SimpleSmooth(LJ93, SimpleSmoothPotential):
    """
    Uses the SimpleSmoothPotential smoothing in combination with LJ93
    """
    name = 'lj9-3simple-smooth'
    def __init__(self, epsilon, sigma, r_c):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε (careful, not work of
                   adhesion in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        r_c     -- emposed cutoff radius
        """
        LJ93.__init__(self, epsilon, sigma, r_c)
        SimpleSmoothPotential.__init__(self, r_c)

    @property
    def r_min(self):
        """
        convenience function returning the location of the energy minimum
        """
        return self._r_min
