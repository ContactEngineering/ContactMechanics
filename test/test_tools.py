#
# Warning: Could no find author name for junge@cmsjunge
# Copyright 2016, 2018, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2015-2016 Till Junge
#           2015 junge@cmsjunge
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
Tests for PyCo helper tools
"""

import unittest

import numpy as np

from ContactMechanics.Tools import evaluate_gradient, mean_err

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="test only serial functionalities, please execute with pytest")


class ToolTest(unittest.TestCase):
    def test_gradient(self):
        coeffs = np.random.random(2) + 1.

        def fun(x):
            return (coeffs * x ** 2).sum()

        def grad(x):
            return 2 * coeffs * x

        x = 20 * (np.random.random(2) - .5)
        tol = 1e-8
        f = fun(x)
        g = grad(x)
        approx_g = evaluate_gradient(fun, x, 1e-5)
        error = mean_err(g, approx_g)

        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))
