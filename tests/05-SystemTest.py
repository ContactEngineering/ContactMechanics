#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   04-SystemTest.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   11 Feb 2015
#
# @brief  Tests the creation of tribosystems
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

import unittest

from PyPyContact.System import System, IncompatibleFormulationError
from PyPyContact.System import IncompatibleResolutionError
import PyPyContact.SolidMechanics as Solid
import PyPyContact.ContactMechanics as Contact
import PyPyContact.Surface as Surface
from numpy.random import rand, random
import numpy as np

class SystemTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        self.radius = 100
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

        self.substrate = Solid.FFTElasticHalfSpace(
            self.res, self.young, self.size)

        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.smooth = Contact.LJ93smooth(self.eps, self.sig, self.gam)

        self.sphere = Surface.Sphere(self.radius, self.res, self.size)


    def test_RejectInconsistentInputTypes(self):
        with self.assertRaises(IncompatibleFormulationError):
            System(12, 13, 24)

    def test_RejectInconsistentSizes(self):
        incompat_res = tuple((2*r for r in self.res))
        incompat_sphere = Surface.Sphere(self.radius, incompat_res, self.size)
        with self.assertRaises(IncompatibleResolutionError):
            System(self.substrate, self.smooth, incompat_sphere)

    def test_SmoothContact(self):
        S = System(self.substrate, self.smooth, self.sphere)
        offset = self.sig
        disp = np.zeros(self.res)
        print(S.evaluate(disp, offset, forces = True))

