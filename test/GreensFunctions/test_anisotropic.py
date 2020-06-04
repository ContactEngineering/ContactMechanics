#
# Copyright 2016, 2019-2020 Lars Pastewka
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

import numpy as np

from ContactMechanics.GreensFunctions import AnisotropicGreensFunction


def test_find_qz_isotropic(tol=1e-5):
    """Test that for an isotropic solid qx^2 + qy^2 = (iqz)^2"""
    C11 = 1
    C44 = 0.3
    gf = AnisotropicGreensFunction(C11, C11 - 2 * C44, C44)
    assert (abs(gf.find_eigenvalues(1, 1) - np.sqrt(2)) < tol).all()
    assert (abs(gf.find_eigenvalues(1, 0.3) -
                np.sqrt(1 + 0.3 ** 2)) < tol).all()


def test_test():
    C11 = 1
    C44 = 0.3
    C12 = C11 - 2 * C44 + 0.1  # 0.3
    gf = AnisotropicGreensFunction(C11, C12, C44)
    Q = gf.find_eigenvalues(1, 0.5)
    # plt.plot(x, [np.linalg.det(gf.bulkop([1, 1, 1j * y])) for y in x],
    # 'r-', lw=4)
    # plt.plot(x, [np.linalg.det(gf.bulkop([-1, -1, 1j * y])) for y in x],
    # 'k-')
    # plt.show()
    gf.find_eigenvectors(1, 0.5, -1j * Q)
