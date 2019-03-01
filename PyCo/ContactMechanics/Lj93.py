#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Lj93.py

@author Till Junge <till.junge@kit.edu>

@date   22 Jan 2015

@brief  9-3 Lennard-Jones potential for wall interactions

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from . import Potential, SmoothPotential
from . import ParabolicCutoffPotential
from . import LinearCorePotential
import numpy as np

class LJ93(Potential):
    """ 9-3 Lennard-Jones potential with optional cutoff radius.

        9-3 Lennard-Jones potential:
        V_l (r) = ε[ 2/15 (σ/r)**9 - (σ/r)**3]

        When used with a cutoff radius, the potential is shifted in order to
        guarantee continuity of forces

        V_lc (r) = V_l (r) - V_l (r_c)
    """

    name = "lj9-3"

    def __init__(self, epsilon, sigma, r_cut=float('inf'),pnp=np):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε
        sigma   -- Lennard-Jones distance parameter σ
        r_cut   -- (default i) optional cutoff radius
        """
        self.eps = epsilon
        self.sig = sigma
        Potential.__init__(self, r_cut,pnp=pnp)

    def __getstate__(self):
        state = super().__getstate__(), self.eps, self.sig
        return state

    def __setstate__(self, state):
        superstate, self.eps, self.sig = state
        super().__setstate__(superstate)


    def __repr__(self, ):
        return ("Potential '{0.name}': ε = {0.eps}, σ = {0.sig}, "
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else '∞')  # nopep8

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

    def __init__(self, epsilon, sigma, gamma=None, r_t=None,pnp=np):
        """
        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε (careful, not work of
                   adhesion in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        gamma   -- (default ε) Work of adhesion, defaults to ε
        r_t     -- (default r_min) transition point, defaults to r_min
        """
        LJ93.__init__(self, epsilon, sigma, None,pnp=pnp)
        SmoothPotential.__init__(self, gamma, r_t)

    def __getstate__(self):
        state = LJ93.__getstate__(self), SmoothPotential.__getstate__(self)
        return state

    def __setstate__(self, state):
        lj93state, smoothpotstate = state
        LJ93.__setstate__(self, lj93state)
        SmoothPotential.__setstate__(self, smoothpotstate)

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', ε = {0.eps}, σ = "
                "{0.sig}{1}{2}").format(
                    self,
                    ", γ = {.gamma}".format(self) if has_gamma else "",  # nopep8
                    ", r_t = {}".format(
                        self.r_t if has_r_t else "r_min"))

    @property
    def r_infl(self):
        """
        convenience function returning the location of the potential's
        inflection point
        Depending on where the transition between LJ93 and spline has been made
        this returns the inflection point of the spline or of the original LJ93
        """
        r_infl_poly = SmoothPotential.get_r_infl_spline(self)

        if r_infl_poly is not None:
            if r_infl_poly < self.r_t:
                return super().r_infl 
                # This is the old property implementation in the LJ93 Potential
            else:
                return r_infl_poly
        else:
            # The Spline wasn't determined already
            return super().r_infl  
            # This is the old property implementation in the LJ93 Potential


def LJ93smoothMin(epsilon, sigma, gamma=None, r_ti=None, r_t_ls=None, pnp = np):
    """
    When starting from a bad guess, or with a bad optimizer, sometimes
    optimisations that include potentials with a singularity at the origin
    fail, because the optimizer chooses a bad step direction and length and
    falls into non-physical territory. This class tries to remedy this by
    replacing the singular repulsive part around zero by a linear function.

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
    return LinearCorePotential(LJ93smooth(epsilon, sigma, gamma, r_t_ls, pnp=pnp), r_ti)

def LJ93SimpleSmooth(epsilon, sigma, r_c, pnp=np):
    """Uses the ParabolicCutoffPotential smoothing in combination with LJ93

        Keyword Arguments:
        epsilon -- Lennard-Jones potential well ε (careful, not work of
                   adhesion in this formulation)
        sigma   -- Lennard-Jones distance parameter σ
        r_c     -- emposed cutoff radius
    """
    return ParabolicCutoffPotential(LJ93(epsilon, sigma, pnp=pnp), r_c)

def LJ93SimpleSmoothMin(epsilon, sigma, r_c, r_ti, pnp=np):
    return LinearCorePotential(LJ93SimpleSmooth(epsilon, sigma, r_c,pnp=pnp), r_ti=r_ti)