#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   AnalysisTest.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   17 Dec 2018

@brief  Tests for PyCo analysis tools; power-spectral density,
        autocorrelation function and variable bandwidth analysis

@section LICENCE

Copyright 2018 Lars Pastewka

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

from PyCo.Topography import UniformNumpyTopography, NonuniformNumpyTopography

from tests.PyCoTest import PyCoTestCase

class PowerSpectrumTest(PyCoTestCase):
    def test_uniform(self):
        for periodic in [True, False]:
            for L in [1.3, 10.6]:
                for k in [2, 4]:
                    for n in [16, 128]:
                        n = 16
                        x = np.arange(n)*L/n
                        h = np.sin(2*np.pi*k*x/L)
                        t = UniformNumpyTopography(h, size=(L,), periodic=periodic)
                        q, C = t.power_spectrum_1D()

                        # The ms height of the sine is 1/2. The sum over the PSD (from -q to +q) is the ms height.
                        # Our PSD only contains *half* of the full PSD (on the +q branch, the -q branch is identical),
                        # therefore the sum over it is 1/4.
                        self.assertAlmostEqual(C.sum()/L, 1/4)

                        if periodic:
                            # The value at the individual wavevector must also equal 1/4. This is only exactly true
                            # for the periodic case. In the nonperiodic, this is convolved with the Fourier transform
                            # of the window function.
                            C /= L
                            r = np.zeros_like(C)
                            r[k] = 1/4
                            self.assertArrayAlmostEqual(C, r)
