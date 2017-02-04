#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   14-Plastic_tests.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   04 Feb 2017

@brief  Tests plastic deformation

@section LICENCE

 Copyright (C) 2015-2017 Till Junge, Lars Pastewka

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

try:
    import numpy as np
    import unittest
    from scipy.optimize import bisect
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.Surface import read, PlasticSurface
    from PyCo.System import SystemFactory
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

# -----------------------------------------------------------------------------
class PlasticTest(unittest.TestCase):
    def test_hard_wall_LBFGS(self):
        # Test that at very low hardness we converge to the bearing area geometry
        surface = read('examples/surface1.out')
        system = SystemFactory(PeriodicFFTElasticHalfSpace(surface.shape, 1.0),
                               HardWall(), PlasticSurface(surface, 0.000000001))
        offset = -0.002
        result = system.minimize_proxy(offset=offset)
        c = result.jac > 0.0
        ncontact = c.sum()

        bearing_area = bisect(lambda x: (surface.profile()>x).sum()-ncontact, -0.03, 0.03)
        cba = surface.profile()>bearing_area

        self.assertTrue(np.logical_not(c == cba).sum() < 11)

if __name__ == '__main__':
    unittest.main()
