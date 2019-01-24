#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Hertz_tests.py

@author Till Junge <till.junge@kit.edu>

@date   05 Oct 2015

@brief  Tests adhesion-free systems for accuracy and compares performance

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
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import make_sphere
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
        for kind in ['ref']: # Add 'opt' to test optimized solver, but does
                             # not work on Travis!
            for nx, ny in [(256, 256), (256, 255), (255, 256)]:
                for disp0, normal_force in [(0.1, None), (0, 15.0)]:
                    sx = 5.0
                    substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s,
                                                        (sx, sx))
                    interaction = HardWall()
                    surface = make_sphere(self.r_s, (nx, ny), (sx, sx))
                    system = SystemFactory(substrate, interaction, surface)

                    result = system.minimize_proxy(offset=disp0,
                                                   external_force=normal_force,
                                                   kind=kind)
                    disp = result.x
                    forces = -result.jac
                    converged = result.success
                    self.assertTrue(converged)

                    if normal_force is not None:
                        self.assertAlmostEqual(normal_force, -forces.sum())
                    normal_force = -forces.sum()
                    a, p0 = radius_and_pressure(normal_force, self.r_s,
                                                self.E_s)

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
