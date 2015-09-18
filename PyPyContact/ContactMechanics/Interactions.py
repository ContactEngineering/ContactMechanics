#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Interaction.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Defines the base class for contact description

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


class Interaction(object):
    "base class for all interactions, e.g. interatomic potentials"
    # pylint: disable=too-few-public-methods
    pass


class HardWall(Interaction):
    "base class for non-smooth contact mechanics"
    # pylint: disable=too-few-public-methods
    pass


class SoftWall(Interaction):
    "base class for smooth contact mechanics"
    def __init__(self):
        self.energy = None
        self.force = None

    def compute(self, gap, pot=True, forces=False, area_scale=1.):
        """
        computes and stores the interaction energy and/or forces based on the
        as fuction of the gap
        Parameters:
        gap        -- array containing the point-wise gap values
        pot        -- (default True) whether the energy should be evaluated
        forces     -- (default False) whether the forces should be evaluated
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        energy, self.force = self.evaluate(
            gap, pot, forces, area_scale)
        self.energy = energy.sum()

    def evaluate(self, gap, pot=True, forces=False, area_scale=1.):
        """
        computes and returns the interaction energy and/or forces based on the
        as fuction of the gap
        Parameters:
        gap        -- array containing the point-wise gap values
        pot        -- (default True) whether the energy should be evaluated
        forces     -- (default False) whether the forces should be evaluated
        area_scale -- (default 1.) scale by this. (Interaction quantities are
                      supposed to be expressed per unit area, so systems need
                      to be able to scale their response for their resolution))
        """
        raise NotImplementedError()
