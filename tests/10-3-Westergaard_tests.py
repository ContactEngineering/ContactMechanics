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
    from PyCo.ReferenceSolutions.Westergaard import _pressure
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
    from PyCo.Surface import NumpySurface
    from PyCo.System import SystemFactory
    from PyCo.Tools.Logger import screen
    from .PyCoTest import PyCoTestCase
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

# -----------------------------------------------------------------------------
class WestergaardTest(PyCoTestCase):
    def setUp(self):
        # system size
        self.sx = 30.0
        self.sy = 1.0
        # equivalent Young's modulus
        self.E_s = 3.56

    def test_constrained_conjugate_gradients(self):
        for kind in ['ref']: # Add 'opt' to test optimized solver, but does
                             # not work on Travis!
            for nx, ny in [(256, 16)]: #, (256, 15), (255, 16)]:
                for disp0, normal_force in [(-0.9, None), (-0.1, None)]: # (0.1, None),
                    substrate = PeriodicFFTElasticHalfSpace((nx, ny), self.E_s,
                                                            (self.sx, self.sy))
                    interaction = HardWall()
                    profile = np.resize(np.cos(2*np.pi*np.arange(nx)/nx), (ny, nx))
                    surface = NumpySurface(profile.T, size=(self.sx, self.sy))
                    system = SystemFactory(substrate, interaction, surface)

                    result = system.minimize_proxy(offset=disp0,
                                                   external_force=normal_force,
                                                   kind=kind)
                    offset = result.offset
                    forces = result.jac
                    displ = result.x[:forces.shape[0], :forces.shape[1]]
                    converged = result.success
                    self.assertTrue(converged)

                    x = np.arange(nx)*self.sx/nx
                    mean_pressure = np.mean(forces)/substrate.area_per_pt
                    pth = mean_pressure * _pressure(x/self.sx, mean_pressure=self.sx*mean_pressure/self.E_s)
                    #import matplotlib.pyplot as plt
                    #plt.figure()
                    ##plt.plot(np.arange(nx)*self.sx/nx, profile)
                    #plt.plot(x, displ[:, 0], 'r-')
                    #plt.plot(x, surface[:, 0]+offset, 'k-')
                    #plt.figure()
                    #plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
                    #plt.plot(x, pth, 'r-')
                    #plt.show()
                    self.assertArrayAlmostEqual(forces[:nx//2, 0]/substrate.area_per_pt, pth[:nx//2], tol=5e-3)

if __name__ == '__main__':
    unittest.main()
