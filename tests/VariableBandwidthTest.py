# -*- coding:utf-8 -*-
"""
@file   VariableBandwidthTests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   06 Sep 2018

@brief  Test tools for variable bandwidth analysis.

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

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

import unittest

import numpy as np

from PyCo.Topography import Topography
from PyCo.Topography.Uniform.VariableBandwidth import checkerboard_tilt_correction
from .PyCoTest import PyCoTestCase

###

class TestAnalysis(PyCoTestCase):

    def test_checkerboard_tilt_correction_2d(self):
        arr = np.zeros([4, 4])
        arr[:2, :2] = 1.0
        outarr = checkerboard_tilt_correction(Topography(arr, arr.shape), (4, 4, 4), (1, 1, 1))
        self.assertArrayAlmostEqual(outarr, np.zeros([4, 4]))

        arr = np.zeros([4, 4])
        arr[:2, :2] = 1.0
        arr[:2, 1] = 2.0
        outarr = checkerboard_tilt_correction(Topography(arr, arr.shape), (4, 4, 4), (1, 1, 1))
        self.assertArrayAlmostEqual(outarr, np.zeros([4, 4]))

###

if __name__ == '__main__':
    unittest.main()
