#
# Copyright 2016, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Tests adhesion-free flat punch results
"""

import unittest
import numpy as np

from ContactMechanics.ReferenceSolutions.Westergaard import _pressure
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import Topography
from ContactMechanics import make_system


class WestergaardTest(unittest.TestCase):
    def setUp(self):
        # system physical_sizes
        self.sx = 30.0
        self.sy = 1.0
        # equivalent Young's modulus
        self.E_s = 3.56

    def test_constrained_conjugate_gradients(self):
        for nx, ny in [(256, 16)]:  # , (256, 15), (255, 16)]:
            for disp0, normal_force in [(-0.9, None),
                                        (-0.1, None)]:  # (0.1, None),
                substrate = PeriodicFFTElasticHalfSpace((nx, ny), self.E_s,
                                                        (self.sx, self.sy))
                profile = np.resize(np.cos(2 * np.pi * np.arange(nx) / nx),
                                    (ny, nx))
                surface = Topography(profile.T, (self.sx, self.sy))
                system = make_system(substrate, surface)

                result = system.minimize_proxy(offset=disp0,
                                               external_force=normal_force,
                                               pentol=1e-9)
                # offset = result.offset
                forces = result.jac
                # displ = result.x[:forces.shape[0], :forces.shape[1]]
                converged = result.success
                self.assertTrue(converged)

                x = np.arange(nx) * self.sx / nx
                mean_pressure = np.mean(forces) / substrate.area_per_pt
                pth = mean_pressure * _pressure(
                    x / self.sx,
                    mean_pressure=self.sx * mean_pressure / self.E_s)

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(np.arange(nx)*self.sx/nx, profile)
                # plt.plot(x, displ[:, 0], 'r-')
                # plt.plot(x, surface[:, 0]+offset, 'k-')
                # plt.figure()
                # plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
                # plt.plot(x, pth, 'r-')
                # plt.show()
                self.assertTrue(np.allclose(
                    forces[:nx // 2, 0] /
                    substrate.area_per_pt, pth[:nx // 2], atol=1e-2))


if __name__ == '__main__':
    unittest.main()
