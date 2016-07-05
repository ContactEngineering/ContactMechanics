#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   10-Hertz_tests.py

@author Till Junge <till.junge@kit.edu>

@date   05 Oct 2015

@brief  Tests adhesion-free systems for accuracy and compares performance

@section LICENCE

 Copyright (C) 2015 Till Junge

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
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Surface import Sphere
    from PyCo.System import SystemFactory
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

# -----------------------------------------------------------------------------
class HertzTest(unittest.TestCase):
    def setUp(self):
        # sphere radius:
        self.r_s = 20.0
        # contact radius
        self.r_c = .2
        # peak pressure
        self.p_0 = 2.5
        # equivalent Young's modulus
        self.E_s = 102.

    def test_elastic_solution(self):
        r = np.linspace(0, self.r_s, 6)/self.r_c
        u = surface_displacements(r) / (self.p_0/self.E_s*self.r_c)
        sig = surface_stress(r)[0]/self.p_0

    def test_constrained_conjugate_gradients(self):
        for nx, ny in [(256, 256), (256, 255), (255, 256)]:
            sx = 5.0
            disp0 = 0.1
            substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s, (sx, sx))
            interaction = HardWall()
            surface = Sphere(self.r_s, (nx, ny), (sx, sx))
            system = SystemFactory(substrate, interaction, surface)

            result = system.minimize_proxy(disp0)
            disp = result.x
            forces = -result.jac
            converged = result.success
            self.assertTrue(converged)

            normal_force = -forces.sum()
            a, p0 = radius_and_pressure(normal_force, self.r_s, self.E_s)

            x = ((np.arange(nx)-nx/2)*sx/nx).reshape(-1,1)
            y = ((np.arange(ny)-ny/2)*sx/ny).reshape(1,-1)
            p_numerical = -forces*(nx*ny/(sx*sx))
            p_analytical = np.zeros_like(p_numerical)
            r = np.sqrt(x**2+y**2)
            p_analytical[r<a] = p0*np.sqrt(1-(r[r<a]/a)**2)

            #import matplotlib.pyplot as plt
            #plt.subplot(1,3,1)
            #plt.pcolormesh(p_analytical-p_numerical)
            #plt.colorbar()
            #plt.plot(x, np.sqrt(self.r_s**2-x**2)-(self.r_s-disp0))
            #plt.subplot(1,3,2)
            #plt.pcolormesh(p_analytical)
            #plt.colorbar()
            #plt.subplot(1,3,3)
            #plt.pcolormesh(p_numerical)
            #plt.colorbar()
            #plt.show()
            msg = ""
            msg +="\np_numerical_type:  {}".format(type(p_numerical))
            msg +="\np_numerical_shape: {}".format(p_numerical.shape)
            msg +="\np_numerical_mean:  {}".format(p_numerical.mean())
            msg +="\np_numerical_dtype: {}".format(p_numerical.dtype)
            msg +="\np_numerical_max:   {}".format(p_numerical.max())
            msg +="\np_analytical_max:  {}".format(p_analytical.max())
            msg +="\nslice_size:        {}".format((r<.99*a).sum())
            msg +="\ncontact_radius a:  {}".format(a)
            msg +="\nnormal_force:      {}".format(normal_force)
            msg +="\n{}".format(result.jac.max()-result.jac.min())

            try:
                self.assertLess(abs(p_analytical[r<0.99*a]-
                                    p_numerical[r<0.99*a]).max()/self.E_s, 1e-3,
                                msg)
            except ValueError as err:
                msg = str(err) + msg
                raise ValueError(msg)


if __name__ == '__main__':
    unittest.main()
