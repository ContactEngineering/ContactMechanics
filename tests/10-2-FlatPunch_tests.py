#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   14-FlatPunch_tests.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   02 Dec 2016

@brief  Tests adhesion-free flat punch results

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

try:
    import unittest
    import numpy as np
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Surface import NumpySurface
    from PyCo.System import SystemFactory
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
                    surface = NumpySurface(
                        np.ma.masked_where(r_sq > self.r_s**2,
                                           np.zeros([nx, ny]))
                        )
                    system = SystemFactory(substrate, interaction, surface)

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
