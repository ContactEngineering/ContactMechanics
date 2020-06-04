#
# Copyright 2016-2017, 2020 Lars Pastewka
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
Tests for PyCo ReferenceSolutions
"""

import unittest

import numpy as np

from scipy.integrate import quad
from scipy.special import ellipk, erf

import ContactMechanics.ReferenceSolutions.GreenwoodTripp as GT
import ContactMechanics.ReferenceSolutions.Hertz as Hz


class ReferenceSolutionsTest(unittest.TestCase):
    def test_Fn(self):
        x = np.linspace(-3, 3, 101)
        np.testing.assert_allclose(GT.Fn(x, 0), (1 - erf(x / np.sqrt(2))) / 2)

    def test_s(self):
        def L(x):
            return np.where(x <= 1, 2 / np.pi * x * ellipk(x ** 2),
                            2 / np.pi * ellipk(1 / x ** 2))

        def snum(ξ):
            return np.array([quad(L, 0, _ξ)[0] for _ξ in ξ])

        x = np.linspace(0.01, 5, 101)
        np.testing.assert_allclose(GT.s(x), snum(x))

    def test_no_oscillating_solution(self):
        μ = 24.0425586841
        w1, p1, rho1 = GT.GreenwoodTripp(0.5, μ)
        w2, p2, rho2 = GT.GreenwoodTripp(1.0, μ)

        self.assertTrue(np.all(np.diff(p1) < 0.0))
        self.assertTrue(np.all(np.diff(p2) < 0.0))

        # import matplotlib.pyplot as plt
        # plt.subplot(1,2,1)
        # plt.plot(rho2, w2, 'k-')
        # plt.plot(rho1, w1, 'r-')
        # plt.subplot(1,2,2)
        # plt.plot(rho2, p2, 'k-')
        # plt.plot(rho1, p1, 'r-')
        # plt.yscale('log')
        # plt.show()

    def test_Hertz_selfconsistency(self):
        Es, R, F = np.random.random(3) * 100
        self.assertAlmostEqual(Hz.normal_load(Hz.penetration(F, R, Es), R, Es),
                               F)
        self.assertAlmostEqual(
            Hz.radius_and_pressure(F, R, Es)[1],
            Hz.max_pressure__penetration(Hz.penetration(F, R, Es), R, Es))

    def test_Hertz_refVals(self):
        "Compare with values computed once"

        Es = 5
        # print("Es {}".format(Es))
        R = 4

        Fprescribed = 7  # N

        np.testing.assert_allclose(Hz.penetration(Fprescribed, R, Es),
                                   0.6507879989531481, rtol=1e-7)
        np.testing.assert_allclose(Hz.radius_and_pressure(Fprescribed, R, Es),
                                   (1.6134286460245437, 1.2839257217043503),
                                   rtol=1e-7)

    def test_Hertz_energy(self):
        Es = 7
        R = 5
        ds = np.linspace(0, 3, 100)
        np.testing.assert_allclose(Hz.elastic_energy(ds[-1], R, Es),
                                   np.trapz(Hz.normal_load(ds, R, Es), x=ds),
                                   rtol=1e-4)
