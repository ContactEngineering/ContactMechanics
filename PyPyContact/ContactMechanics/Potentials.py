#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Potential.py

@author Till Junge <till.junge@kit.edu>

@date   21 Jan 2015

@brief  Generic potential class, all potentials inherit it

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

from .Interactions import SoftWall


class Potential(SoftWall):
    """ Describes the minimum interface to interaction potentials for
        PyPyContact. These pontentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
    # pylint: disable=abstract-class-not-used
    name = "generic_potential"

    class PotentialError(Exception):
        # pylint: disable=missing-docstring
        pass

    def __init__(self, ):
        super().__init__()
        self.r_c = None
        self.curb = None

    def __repr__(self):
        return ("Potential '{0.name}', cut-off radius r_cut = " +
                "{0.r_c}").format(self)

    def compute(self, gap, pot=True, forces=False, curb=False):
        # pylint: disable=arguments-differ
        energy, self.force, self.curb = self.evaluate(
            gap, pot, forces, curb)
        self.energy = energy.sum()

    def evaluate(self, gap, pot=True, forces=False, curb=False):
        # pylint: disable=arguments-differ
        raise NotImplementedError()
