#
# Copyright 2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import pytest
import unittest
import numpy as np


from PyCo.Topography import Topography, NonuniformLineScan

from NuMPI import MPI
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")

class SinewaveTestUniform(unittest.TestCase):
    def setUp(self):
        n = 256
        X, Y = np.mgrid[slice(0,n),slice(0,n)]

        self.hm = 0.1
        self.L = n
        self.sinsurf = np.sin(2 * np.pi / self.L * X) * np.sin(2 * np.pi / self.L * Y) * self.hm
        self.size= (self.L,self.L)

        self.surf = Topography(self.sinsurf, physical_sizes=self.size)

        self.precision = 5

    def test_rms_curvature(self):
        numerical = self.surf.rms_curvature()
        analytical = np.sqrt(16*np.pi**4 *self.hm**2 / self.L**4 )
        #print(numerical-analytical)
        self.assertAlmostEqual(numerical,analytical,self.precision)

    def test_rms_slope(self):
        numerical = self.surf.rms_slope()
        analytical = np.sqrt(2*np.pi ** 2 * self.hm**2 / self.L**2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height(self):
        numerical = self.surf.rms_height()
        analytical = np.sqrt(self.hm**2 / 4)

        self.assertEqual(numerical,analytical)


class SinewaveTestNonuniform(unittest.TestCase):
    def setUp(self):
        n = 256

        self.hm = 0.1
        self.L = n
        self.X = np.arange(n+1)  # n+1 because we need the endpoint
        self.sinsurf = np.sin(2 * np.pi * self.X / self.L) * self.hm

        self.precision = 5

#    def test_rms_curvature(self):
#        numerical = Nonuniform.rms_curvature(self.X, self.sinsurf)
#        analytical = np.sqrt(16*np.pi**4 *self.hm**2 / self.L**4 )
#        #print(numerical-analytical)
#        self.assertAlmostEqual(numerical,analytical,self.precision)

    def test_rms_slope(self):
        numerical = NonuniformLineScan(self.X, self.sinsurf).rms_slope()
        analytical = np.sqrt(2*np.pi ** 2 * self.hm**2 / self.L**2)
        # print(numerical-analytical)
        self.assertAlmostEqual(numerical, analytical, self.precision)

    def test_rms_height(self):
        numerical = NonuniformLineScan(self.X, self.sinsurf).rms_height()
        analytical = np.sqrt(self.hm**2 / 2)
        #numerical = np.sqrt(np.trapz(self.sinsurf**2, self.X))

        self.assertAlmostEqual(numerical, analytical, self.precision)

if __name__ == '__main__':
    unittest.main()
