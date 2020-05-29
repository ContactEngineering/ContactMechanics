#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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
Tests for PyCo ReferenceSolutions
"""

try:
    import unittest
    import numpy as np
    import warnings
    from scipy.integrate import quad
    from scipy.special import ellipk, erf

    from tests.Topography.PyCoTest import PyCoTestCase
    import PyCo.Adhesion.ReferenceSolutions.MaugisDugdale as MD
    import PyCo.ContactMechanics.ReferenceSolutions.Hertz as Hz

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class ReferenceSolutionsTest(PyCoTestCase):
    def test_md_dmt_limit(self):
        A = np.linspace(0.001, 10, 11)
        N, d = MD._load_and_displacement(A, 1e-12)
        self.assertArrayAlmostEqual(N, A**3-2)
        self.assertArrayAlmostEqual(d, A**2, tol=1e-5)
    def test_md_jkr_limit(self):
        A = np.linspace(0.001, 10, 11)
        N, d = MD._load_and_displacement(A, 1e3)
        self.assertArrayAlmostEqual(N, A**3-A*np.sqrt(6*A), tol=1e-4)
        self.assertArrayAlmostEqual(d, A**2-2/3*np.sqrt(6*A), tol=1e-4)


