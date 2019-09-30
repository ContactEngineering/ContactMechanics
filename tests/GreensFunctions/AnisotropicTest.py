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

import numpy as np

import pytest

from PyCo.SolidMechanics.GreensFunctions import AnisotropicGF

import matplotlib.pyplot as plt


def test_isotropic(tol=1e-6):
    """Test that for an isotropic solid qx^2 + qy^2 = (iqz)^2"""
    C11 = 1
    C12 = 0.5
    C44 = 0.3
    gf = AnisotropicGF(1, C11 - 2 * C44, 0.3)
    assert abs(gf.find_qz(1, 1) - np.sqrt(2)) < tol
    assert abs(gf.find_qz(1, 0.3) - np.sqrt(1 + 0.3 ** 2)) < tol


def test_test():
    C11 = 1
    C12 = 0.5
    C44 = 0.3
    gf = AnisotropicGF(1, C11 - 2 * C44, 0.3)
    x = np.linspace(-2, 2, 101)
    plt.plot(x, [np.linalg.det(gf.bulkop([1, 1, 1j * y])) for y in x], 'k-')
    plt.show()
    print(gf.bulkop([1, 1, 0]))
    gf.find_qz(1, 1)
