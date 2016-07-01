#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Harmonic.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   06 Jun 2016

@brief  Harmonic potential for wall interaction

@section LICENCE

 Copyright (C) 2016 Lars Pastewka

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


class HarmonicPotential(Potential):
    """ Repulsive harmonic potential.

        Harmonic potential:
        V (r) = 1/2 k r**2 for r < 0
    """

    name = "lj9-3"

    def __init__(self, spring_constant):
        """
        Keyword Arguments:
        spring_constant -- Spring constant k
        """
        self.spring_constant = spring_constant
        Potential.__init__(self, 0)

    def __repr__(self, ):
        return ("Potential '{0.name}': k = {0.spring_constant}").format(self)

    @property
    def r_min(self):
        """convenience function returning the location of the energy minimum"""
        return 0

    @property
    def r_infl(self):
        """convenience function returning the location of the potential's
        inflection point (if applicable)
        """
        return None

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=invalid-name
        V = dV = ddV = None
        if pot:
            V = 0.5*self.spring_constant*r**2
        if forces:
            # Forces are the negative gradient
            dV = -self.spring_constant*r
        if curb:
            ddV = self.spring_constant
        return (V, dV, ddV)
