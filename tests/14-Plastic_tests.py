#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   14-Plastic_tests.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   04 Feb 2017

@brief  Tests plastic deformation

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
        # Test that at very low hardness we converge to (almost) the bearing
        # area geometry
        surface = read('examples/surface1.out')
        system = SystemFactory(PeriodicFFTElasticHalfSpace(surface.shape, 1.0),
                               HardWall(), PlasticSurface(surface, 0.0000000001))
        offset = -0.002
        result = system.minimize_proxy(offset=offset)
        c = result.jac > 0.0
        ncontact = c.sum()

        bearing_area = bisect(lambda x: (surface.profile()>x).sum()-ncontact, -0.03, 0.03)
        cba = surface.profile()>bearing_area

        self.assertTrue(np.logical_not(c == cba).sum() < 25)

if __name__ == '__main__':
    unittest.main()
