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
import math
import numpy as np


class Potential(SoftWall):
    """ Describes the minimum interface to interaction potentials for
        PyPyContact. These pontentials are purely 1D, which allows for a few
        simplifications. For instance, the potential energy and forces can be
        computed at any point in the problem from just the one-dimensional gap
        (h(x,y)-z(x,y)) at that point
    """
    name = "generic_potential"

    class PotentialError(Exception):
        "umbrella exception for potential-related issues"
        pass

    class SliceableNone(object):
        """small helper class to remedy numpy's lack of views on
        index-sliced array views. This construction avoid the computation
        of all interactions as with np.where, and copies"""
        # pylint: disable=too-few-public-methods
        __slots__ = ()

        def __setitem__(self, index, val):
            pass

        def __getitem__(self, index):
            pass

    def __init__(self, r_cut):
        super().__init__()
        self.r_c = r_cut
        if r_cut is not None:
            self.has_cutoff = not math.isinf(self.r_c)
        else:
            self.has_cutoff = False
        if self.has_cutoff:
            self.offset = self.naive_pot(self.r_c)[0]
        else:
            self.offset = 0

        self.curb = None

    def __repr__(self):
        return ("Potential '{0.name}', cut-off radius r_cut = " +
                "{0.r_c}").format(self)

    def compute(self, gap, pot=True, forces=False, curb=False):
        # pylint: disable=arguments-differ
        energy, self.force, self.curb = self.evaluate(
            gap, pot, forces, curb)
        self.energy = energy.sum()

    def naive_pot(self, r, pot=True, forces=False, curb=False):
        """ Evaluates the potential and its derivatives without cutoffs or
            offsets.
            Keyword Arguments:
            r      -- array of distances
            pot    -- (default True) if true, returns potential energy
            forces -- (default False) if true, returns forces
            curb   -- (default False) if true, returns second derivative

        """
        raise NotImplementedError()

    def evaluate(self, r, pot=True, forces=False, curb=False):
        """Evaluates the potential and its derivatives
        Keyword Arguments:
        r      -- array of distances
        pot    -- (default True) if true, returns potential energy
        forces -- (default False) if true, returns forces
        curb   -- (default False) if true, returns second derivative
        """
        # pylint: disable=bad-whitespace
        # pylint: disable=arguments-differ
        r = np.array(r)
        if r.shape == ():
            r.shape = (1, )
        inside_slice = r < self.r_c
        V = np.zeros_like(r) if pot else self.SliceableNone()
        dV = np.zeros_like(r) if forces else self.SliceableNone()
        ddV = np.zeros_like(r) if curb else self.SliceableNone()

        V[inside_slice], dV[inside_slice], ddV[inside_slice] = self.naive_pot(
            r[inside_slice], pot, forces, curb)
        if V[inside_slice] is not None:
            V[inside_slice] -= self.offset
        return (V if pot else None,
                dV if forces else None,
                ddV if curb else None)

    @property
    def r_min(self):
        """
        convenience function returning the location of the enery minimum
        """
        raise NotImplementedError()

    @property
    def naive_min(self):
        """ convenience function returning the energy minimum of the bare
           potential

        """
        return self.naive_pot(self.r_min)[0]
