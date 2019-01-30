#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   14-FlatPunch_tests.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   02 Dec 2016

@brief  Tests adhesion-free flat punch results

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
    import unittest
    import numpy as np
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import Topography
    from PyCo.System import make_system
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

# -----------------------------------------------------------------------------
class FlatPunchTest(unittest.TestCase):
    def setUp(self):
        # punch radius:
        self.r_s = 20.0
        # equivalent Young's modulus
        self.E_s = 3.56

    def test_constrained_conjugate_gradients(self):
        for kind in ['ref']: # Add 'opt' to test optimized solver, but does
                             # not work on Travis!
            for nx, ny in [(256, 256), (256, 255), (255, 256)]:
                for disp0, normal_force in [(0, 15.0)]: # (0.1, None),
                    sx = sy = 2.5*self.r_s
                    substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s,
                                                        (sx, sy))
                    interaction = HardWall()
                    r_sq = (sx/nx*(np.arange(nx)-nx//2)).reshape(-1,1)**2 + \
                           (sy/ny*(np.arange(ny)-ny//2)).reshape(1,-1)**2
                    surface = Topography(
                        np.ma.masked_where(r_sq > self.r_s**2,
                                           np.zeros([nx, ny])),
                        (nx, ny)
                        )
                    system = make_system(substrate, interaction, surface)

                    result = system.minimize_proxy(offset=disp0,
                                                   external_force=normal_force,
                                                   kind=kind,
                                                   pentol=1e-4)
                    offset = result.offset
                    forces = -result.jac
                    converged = result.success
                    self.assertTrue(converged)

                    # Check contact stiffness
                    self.assertAlmostEqual(-forces.sum()/offset / (2*self.r_s*self.E_s),
                                           1.0, places=2)

if __name__ == '__main__':
    unittest.main()
