import unittest


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

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

if _withMPI:
    from FFTEngine import PFFTEngine
    from FFTEngine.helpers import gather
    from MPITools.Tools.ParallelNumpy import ParallelNumpy

try:
    import unittest
    import numpy as np
    from PyCo.ContactMechanics import HardWall
    from PyCo.ReferenceSolutions.Westergaard import _pressure
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
    from PyCo.Topography import UniformNumpyTopography
    from PyCo.System import make_system
    from PyCo.Tools.Logger import screen
    from .PyCoTest import PyCoTestCase

    from PyCo.Tools.Logger import Logger

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

        self.comm = MPI.COMM_WORLD

        self.pnp = ParallelNumpy(self.comm)


    def test_constrained_conjugate_gradients(self):
        for kind in ['ref']: # Add 'opt' to test optimized solver, but does
                             # not work on Travis!
            for nx, ny in [(256, 16)]: #, (256, 15), (255, 16)]: #256,16
                for disp0, normal_force in [(-0.9, None), (-0.1, None)]: # (0.1, None),
                    substrate = PeriodicFFTElasticHalfSpace((nx, ny), self.E_s,
                                                            (self.sx, self.sy),fftengine=PFFTEngine((nx,ny),self.comm))
                    interaction = HardWall()
                    profile = np.resize(np.cos(2*np.pi*np.arange(nx)/nx), (ny, nx))
                    surface =UniformNumpyTopography(profile.T, size=(self.sx, self.sy),
                                                    #resolution=substrate.resolution,
                                                    subdomain_location = substrate.topography_subdomain_location,
                                                    subdomain_resolution = substrate.topography_subdomain_resolution,
                                                    pnp=substrate.pnp)
                    system = make_system(substrate, interaction, surface)

                    result = system.minimize_proxy(offset=disp0,
                                                   external_force=normal_force,
                                                   kind=kind,
                                                   pentol=1e-9)
                    offset = result.offset
                    forces = result.jac
                    displ = result.x[:forces.shape[0], :forces.shape[1]]
                    converged = result.success
                    self.assertTrue(converged)

                    #print(forces)
                    #print(displ)

                    x = np.arange(nx)*self.sx/nx
                    mean_pressure = self.pnp.sum(forces)/np.prod(substrate.size)
                    pth = mean_pressure * _pressure(x/self.sx, mean_pressure=self.sx*mean_pressure/self.E_s)

                    # symetrize the Profile
                    pth[1:] = pth[1:] + pth[:0:-1]

                    #import matplotlib.pyplot as plt
                    #plt.figure()
                    ##plt.plot(np.arange(nx)*self.sx/nx, profile)
                    #plt.plot(x, displ[:, 0], 'r-')
                    #plt.plot(x, surface[:, 0]+offset, 'k-')
                    #plt.figure()
                    #plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
                    #plt.plot(x, pth, 'r-')
                    #plt.show()
                    self.assertArrayAlmostEqual(forces[:, 0]/substrate.area_per_pt, pth[substrate.subdomain_slice[0]], tol=1e-2)

suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(WestergaardTest)])
if __name__ in  ['__main__','builtins']:
    print("Running unittest MPI_FileIO_Test")
    result = unittest.TextTestRunner().run(suite)