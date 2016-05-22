#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Lj93.py

@author Till Junge <till.junge@kit.edu>

@date   22 Jan 2015

@brief  9-3 Lennard-Jones potential for wall interactions

@section LICENCE

 Copyright (C) 2015 Till Junge

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from . import Potential
import numpy as np

class ExpPotential(Potential):
    """ V(g) = -gamma0*e^(-g(r)/rho)
    """

    name = "adh"

    def __init__(self, gamma0,rho,r_cut=float('inf')):
        """
        Keyword Arguments:
        gamma0 -- surface energy at perfect contact
        rho   -- attenuation length
        """
        self.rho = rho
        self.gam = gamma0
        Potential.__init__(self,r_cut)


    def __repr__(self, ):
        return ("Potential '{0.name}': eps = {0.eps}, sig = {0.sig},"
                "r_c = {1}").format(
                    self, self.r_c if self.has_cutoff else 'None')

    @property
    def r_min(self):
        return None

    @property
    def r_infl(self):
        return None

    def naive_pot(self, r,pot=True,forces=False,curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets. These have been collected in a single method to reuse the
            computated LJ terms for efficiency
            V(g) = -gamma0*e^(-g(r)/rho)
            V'(g) = (gamma0/rho)*e^(-g(r)/rho)
            V''(g) = -(gamma0/r_ho^2)*e^(-g(r)/rho)

            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        g = -r/self.rho

        # Use exponential only for r > 0
        m = g < 0.0
        V = np.zeros_like(g)
        dV = np.zeros_like(g)
        ddV = np.zeros_like(g)
        V[m] = -self.gam*np.exp(g[m])
        dV[m] = V[m]/self.rho
        ddV[m] = V[m]/self.rho**2

        # Linear function for r < 0. This avoid numerical overflow at small r.
        m = np.logical_not(m)
        V[m] = -self.gam*(1+g[m]+0.5*g[m]**2)
        dV[m] = -self.gam/self.rho*(1+g[m])
        ddV[m] = -self.gam/self.rho**2

        return V, dV, ddV
