#
# Copyright 2019 Lars Pastewka
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
Tests for power-spectral density analysis
"""

import unittest

import numpy as np

from PyCo.Topography import UniformLineScan, NonuniformLineScan
from PyCo.Topography.Generation import fourier_synthesis
from PyCo.Topography.Nonuniform.PowerSpectrum import sinc, dsinc

from ..PyCoTest import PyCoTestCase


class PowerSpectrumTest(PyCoTestCase):
    def test_uniform(self):
        for periodic in [True, False]:
            for L in [1.3, 10.6]:
                for k in [2, 4]:
                    for n in [16, 128]:
                        x = np.arange(n) * L / n
                        h = np.sin(2 * np.pi * k * x / L)
                        t = UniformLineScan(h, size=L, periodic=periodic)
                        q, C = t.power_spectrum_1D()

                        # The ms height of the sine is 1/2. The sum over the PSD (from -q to +q) is the ms height.
                        # Our PSD only contains *half* of the full PSD (on the +q branch, the -q branch is identical),
                        # therefore the sum over it is 1/4.
                        self.assertAlmostEqual(C.sum() / L, 1 / 4)

                        if periodic:
                            # The value at the individual wavevector must also equal 1/4. This is only exactly true
                            # for the periodic case. In the nonperiodic, this is convolved with the Fourier transform
                            # of the window function.
                            C /= L
                            r = np.zeros_like(C)
                            r[k] = 1 / 4
                            self.assertArrayAlmostEqual(C, r)

    @unittest.skip
    def test_nonuniform_on_uniform_grid(self):
        for L in [1.3, 10.6]:
            for k in [2, 8]:
                for n in [64]:
                    x = np.arange(n + 1) * L / n
                    h = np.sin(2 * np.pi * k * x / L)
                    t = NonuniformLineScan(x, h)

                    pad = 64
                    i = t.interpolate(512, padding=4096)
                    qi, Ci = i.power_spectrum_1D(window='None')
                    Ci *= pad

                    q, C = t.power_spectrum_1D(q=qi, window='None')

                    import matplotlib.pyplot as plt
                    plt.plot(*t.positions_and_heights(), lw=4)
                    plt.plot(*i.positions_and_heights())
                    plt.show()

                    plt.plot(q[100:], C[100:], label='true')
                    plt.plot(qi[100:], Ci[100:], label='interpolated')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.legend(loc='best')
                    plt.show()

                    # Throw out the high frequency data points
                    maxi = len(q)//8
                    qi = qi[:maxi]
                    Ci = Ci[:maxi]
                    q = q[:maxi]
                    C = C[:maxi]

                    # Throw out data points below a certain numerical threshold
                    m = C > 1e-5
                    qi = qi[m]
                    Ci = Ci[m]
                    q = q[m]
                    C = C[m]

                    self.assertArrayAlmostEqual(C, Ci, tol=0.1)
                    self.assertAlmostEqual(C.sum()/Ci.sum()-1, 0, places=1)

    def test_invariance(self):
        for a, b, c in [(2.3, 1.2, 1.7),
                        (1.5, 3.1, 3.1),
                        (0.5, 1.0, 1.0),
                        (0.5, -0.5, 0.5)]:
            q = np.linspace(0.0, 2*np.pi/a, 101)

            x = np.array([-a, a])
            h = np.array([b, c])
            _, C1 = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            x = np.array([-a, 0, a])
            h = np.array([b, (b+c)/2, c])
            _, C2 = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            x = np.array([-a, 0, a/2, a])
            h = np.array([b, (b+c)/2, (3*c+b)/4, c])
            _, C3 = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            self.assertArrayAlmostEqual(C1, C2)
            self.assertArrayAlmostEqual(C2, C3)

    def test_rectangle(self):
        for a, b in [(2.3, 1.45), (10.2, 0.1)]:
            x = np.array([-a, a])
            h = np.array([b, b])

            q = np.linspace(0.01, 8 * np.pi / a, 101)

            q, C = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            C_ana = (2 * b * np.sin(a * q) / q) ** 2
            C_ana /= 2 * a

            self.assertArrayAlmostEqual(C, C_ana)

    def test_triangle(self):
        for a, b in [(0.5, -0.5), (1, 1), (2.3, 1.45), (10.2, 0.1)]:
            x = np.array([-a, a])
            h = np.array([-b, b])

            q = np.linspace(0.01, 8 * np.pi / a, 101)

            _, C = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            C_ana = (2 * b * (a * q * np.cos(a * q) - np.sin(a * q)) / (a * q ** 2)) ** 2
            C_ana /= 2 * a

            self.assertArrayAlmostEqual(C, C_ana)

    def test_rectangle_and_triangle(self):
        for a, b, c, d in [(0.123, 1.45, 10.1, 9.3),
                           (-0.1, 5.4, -0.1, 3.43),
                           (-1, 1, 1, 1)]:
            x = np.array([a, b])
            h = np.array([c, d])

            q = np.linspace(0.01, 8 * np.pi / (b-a), 101)

            q, C = NonuniformLineScan(x, h).power_spectrum_1D(q=q, window='None')

            C_ana = np.exp(-1j * (a + b) * q) * (
                        np.exp(1j * a * q) * (c - d + 1j * (a - b) * d * q) +
                        np.exp(1j * b * q) * (d - c - 1j * (a - b) * c * q)
                        ) / ((a - b) * q ** 2)
            C_ana = np.abs(C_ana) ** 2 / (b - a)

            self.assertArrayAlmostEqual(C, C_ana)

    def test_dsinc(self):
        self.assertAlmostEqual(dsinc(0), 0)
        self.assertAlmostEqual(dsinc(np.pi)*np.pi, -1)
        self.assertAlmostEqual(dsinc(2*np.pi)*np.pi, 1 / 2)
        self.assertAlmostEqual(dsinc(3*np.pi)*np.pi, -1 / 3)
        self.assertArrayAlmostEqual(dsinc([0, np.pi])*np.pi, [0, -1])
        self.assertArrayAlmostEqual(dsinc([0, 2*np.pi])*np.pi, [0, 1 / 2])
        self.assertArrayAlmostEqual(dsinc([0, 3*np.pi])*np.pi, [0, -1 / 3])

        dx = 1e-9
        for x in [0, 0.5e-6, 1e-6, 0.5, 1]:
            v1 = sinc(x + dx)
            v2 = sinc(x - dx)
            self.assertAlmostEqual(dsinc(x), (v1 - v2) / (2 * dx), places=5, msg='x = {}'.format(x))

    def test_NaNs(self):
        surf = fourier_synthesis([1024, 512], [2, 1], 0.8, rms_slope=0.1)
        q, C = surf.power_spectrum_2D(nbins=1000)
        self.assertEqual(np.isnan(C).sum(), 0)