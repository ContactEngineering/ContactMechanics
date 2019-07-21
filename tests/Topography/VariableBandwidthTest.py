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
"""
Test tools for variable bandwidth analysis.
"""

import unittest

import numpy as np

from PyCo.Topography import Topography, UniformLineScan
from PyCo.Topography.Generation import fourier_synthesis
from ..PyCoTest import PyCoTestCase
import pytest
from NuMPI import MPI
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")
###

class TestVariableBandwidth(PyCoTestCase):

    def test_checkerboard_detrend_1d(self):
        arr = np.zeros([4])
        arr[:2] = 1.0
        outarr = UniformLineScan(arr, arr.shape).checkerboard_detrend((2, ))
        self.assertArrayAlmostEqual(outarr, np.zeros([4]))

    def test_checkerboard_detrend_2d(self):
        arr = np.zeros([4, 4])
        arr[:2, :2] = 1.0
        outarr = Topography(arr, arr.shape).checkerboard_detrend((2, 2))
        self.assertArrayAlmostEqual(outarr, np.zeros([4, 4]))

        arr = np.zeros([4, 4])
        arr[:2, :2] = 1.0
        arr[:2, 1] = 2.0
        outarr = Topography(arr, arr.shape).checkerboard_detrend((2, 2))
        self.assertArrayAlmostEqual(outarr, np.zeros([4, 4]))

    def test_checkerboard_detrend_with_no_subdivisions(self):
        r = 32
        x, y = np.mgrid[:r, :r]
        h = 1.3*x - 0.3*y + 0.02*x*x + 0.03*y*y - 0.013*x*y
        t = Topography(h, (1, 1), periodic=False)
        # This should be the same as a detrend with detrend_mode='height'
        ut1 = t.checkerboard_detrend((1, 1))
        ut2 = t.detrend().heights()
        self.assertArrayAlmostEqual(ut1, ut2)

    def test_self_affine_topography_1d(self):
        r = 16384
        for H in [0.3, 0.8]:
            t0 = fourier_synthesis((r, ), (1, ), H, rms_slope=0.1,
                                   amplitude_distribution=lambda n: 1.0)

            for t in [t0, t0.to_nonuniform()]:
                mag, bwidth, rms = t.variable_bandwidth(nb_grid_pts_cutoff=r//32)
                self.assertAlmostEqual(rms[0], t.detrend().rms_height())
                self.assertArrayAlmostEqual(bwidth, t.physical_sizes[0] / mag)
                # Since this is a self-affine surface, rms(mag) ~ mag^-H
                b, a = np.polyfit(np.log(mag[1:]), np.log(rms[1:]), 1)
                # The error is huge...
                self.assertTrue(abs(H+b) < 0.1)

    def test_self_affine_topography_2d(self):
        r = 2048
        res = [r, r]
        for H in [0.3, 0.8]:
            t = fourier_synthesis(res, (1, 1), H, rms_slope=0.1,
                                  amplitude_distribution=lambda n: 1.0)
            mag, bwidth, rms = t.variable_bandwidth(nb_grid_pts_cutoff=r//32)
            self.assertAlmostEqual(rms[0], t.detrend().rms_height())
            # Since this is a self-affine surface, rms(mag) ~ mag^-H
            b, a = np.polyfit(np.log(mag[1:]), np.log(rms[1:]), 1)
            # The error is huge...
            self.assertTrue(abs(H+b) < 0.1)

###

if __name__ == '__main__':
    unittest.main()
