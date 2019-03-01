#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   VdW82.py

@author Till Junge <till.junge@kit.edu>

@date   25 Feb 2015

@brief  Implements a van der Waals-type interaction as discribed in
        http://dx.doi.org/10.1103/PhysRevLett.111.035502

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
import numpy as np

from . import Potential, SmoothPotential
from . import ParabolicCutoffPotential
from . import LinearCorePotential


class VDW82(Potential):
    """
    Van der Waals-type attractive potential with a fantasy repulsive model
    (like Lennard-Jones). The potential uses the formulation of Lessel et al.
    2013 (http://dx.doi.org/10.1103/PhysRevLett.111.035502). However, the oxide
    layer is supposed do be thick
    """
    name = 'v-d-Waals82'

    def __init__(self, c_sr, hamaker, r_cut=float('inf'),pnp=np):
        """
        Keyword Arguments:
        c_sr            -- coefficient for repulsive part
        hamaker         -- Hamaker constant for substrate
        r_cut           -- (default +∞) optional cutoff radius
        """
        self.c_sr = c_sr
        self.hamaker = hamaker
        Potential.__init__(self, r_cut, pnp=pnp)

    def __getstate__(self):
        state = super().__getstate__(), self.c_sr, self.hamaker
        return state

    def __setstate__(self, state):
        superstate, self.c_sr, self.hamaker = state
        super().__setstate__(superstate)

    def __repr__(self, ):
        return ("Potential '{0.name}': C_SR = {0.c_sr}, A_l = {0.hamaker}, "
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else '∞')  # nopep8

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated vdW terms for efficiency

                           A      C_sr
            vdW(r)  = - ─────── + ────
                            2       8
                        12⋅r ⋅π    r

                        A      8⋅C_sr
            vdW'(r) = ────── - ──────
                         3        9
                      6⋅r ⋅π     r

                          A      72⋅C_sr
            vdW"(r) = - ────── + ───────
                           4        10
                        2⋅r ⋅π     r
            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        V = dV = ddV = None
        r_2 = r**-2
        c_sr_r6 = self.c_sr*r_2**3
        a_pi = self.hamaker/np.pi

        if pot:
            V = r_2*(-a_pi/12 + c_sr_r6)
        if forces or curb:
            r_3 = r_2/r
        if forces:
            # Forces are the negative gradient
            dV = r_3*(-a_pi/6 + 8*c_sr_r6)
        if curb:
            ddV = r_3/r*(-a_pi/2 + 72*c_sr_r6)
        return (V, dV, ddV)

    @property
    def r_min(self):
        """convenience function returning the location of the enery minimum
                               ________
                 2/3 6 ___    ╱ C_sr⋅π
        r_min = 2   ⋅╲╱ 3 ⋅6 ╱  ──────
                           ╲╱     A
        """
        return 2**(2./3)*3**(1./6)*(self.c_sr*np.pi/self.hamaker)**(1./6)

    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)

                            ________
                 3 ____    ╱ C_sr⋅π
        r_infl = ╲╱ 12 ⋅6 ╱  ──────
                        ╲╱     A
        """
        return (144*np.pi * self.c_sr / self.hamaker)**(1./6.)


class VDW82smooth(VDW82, SmoothPotential):
    """
    Van der Waals potential with a smoothly finite tail, see docstring of
    SmoothPotential
    """
    name = 'vdw82smooth'

    def __init__(self, c_sr, hamaker, gamma=None, r_t=None,pnp=np):
        """
        Keyword Arguments:
        c_sr    -- coefficient for repulsive part
        hamaker -- Hamaker constant for substrate
        gamma   -- (default ε) Work of adhesion, defaults to ε
        r_t     -- (default r_min) transition point, defaults to r_min
        """
        VDW82.__init__(self, c_sr, hamaker,pnp=pnp)
        SmoothPotential.__init__(self, gamma, r_t)

    def __repr__(self):
        has_gamma = -self.gamma != self.naive_min
        has_r_t = self.r_t != self.r_min
        return ("Potential '{0.name}', C_sr = {0.c_sr}, Hamaker = "
                "{0.hamaker}{1}{2}").format(
                    self,
                    ", γ = {.gamma}".format(self) if has_gamma else "",
                    ", r_t = {}".format(
                        self.r_t if has_r_t else "r_min"))  # nopep8

    @property
    def r_infl(self):
        """
        convenience function returning the location of the potential's
        inflection point
        Depending on where the transition between VDW82 and spline has been made
        this returns the inflection point of the spline or of the original 
        VDW82 Potential
        """
        r_infl_poly = SmoothPotential.get_r_infl_spline(self)
        if r_infl_poly is not None:
            if r_infl_poly < self.r_t:
                return super().r_infl
                # This is the old property implementation in the VDW82 Potential
            else:
                return r_infl_poly
        else:
            # The Spline wasn't determined already
            return super().r_infl
            # This is the old property implementation in the VDW82 Potential

def VDW82smoothMin(c_sr, hamaker, gamma=None, r_ti=None, r_t_ls=None, pnp=np):
    """
    When starting from a bad guess, or with a bad optimizer, sometimes
    optimisations that include potentials with a singularity at the origin
    fail, because the optimizer chooses a bad step direction and length and
    falls into non-physical territory. This class tries to remedy this by
    replacing the singular repulsive part around zero by a linear function.

    Keyword Arguments:
    c_sr    -- coefficient for repulsive part
    hamaker -- Hamaker constant for substrate
    gamma   -- (default ε) Work of adhesion, defaults to ε
    r_ti    -- (default r_min/2) transition point between linear function
               and lj, defaults to r_min
    r_t_ls  -- (default r_min) transition point between lj and spline,
                defaults to r_min
    """
    return LinearCorePotential(VDW82smooth(c_sr, hamaker, gamma, r_t_ls, pnp=pnp), r_ti)

def VDW82SimpleSmooth(c_sr, hamaker, r_c, pnp=np):
    """Uses the ParabolaCutoffPotential smoothing in combination with VDW82

        Keyword Arguments:
        c_sr    -- coefficient for repulsive part
        hamaker -- Hamaker constant for substrate
        r_c     -- emposed cutoff radius
    """
    return ParabolicCutoffPotential(VDW82(c_sr, hamaker, pnp=pnp), r_c)

def VDW82SimpleSmoothMin(c_sr, hamaker, r_c, r_ti, pnp=np):
    return LinearCorePotential(VDW82SimpleSmooth(c_sr, hamaker, r_c, pnp=pnp), r_ti=r_ti)