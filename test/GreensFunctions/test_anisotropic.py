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
import pytest

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


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def test_scalar_wavevector():
    gf = AnisotropicGreensFunction(1.0, 0.5, 0.3)
    np.testing.assert_allclose(gf.stiffness(0.3, 0.1), gf.stiffness(np.array([0.3]), np.array([0.1]))[0])
    np.testing.assert_allclose(gf.greens_function(0.3, 0.1),
                               gf.greens_function(np.array([0.3]), np.array([0.1]))[0])


@pytest.mark.parametrize('thickness', [None, 2.0])
@pytest.mark.parametrize('axis, angle', [([1, 0, 0], np.pi / 6), ([1, 2, 3], 0.7)])
def test_rotated_crystal(axis, angle, thickness):
    # These orientations break the z -> -z symmetry of the elastic constants;
    # det(M) then has odd powers of qz, which were neglected
    qx = np.array([1.0, 0.3, -0.7, 2.0])
    qy = np.array([0.0, 0.8, 0.4, -1.5])
    gf = AnisotropicGreensFunction(1.0, 0.5, 0.3, thickness=thickness, R=_rotation(axis, angle))
    for K, G in zip(gf.stiffness(qx, qy), gf.greens_function(qx, qy)):
        np.testing.assert_allclose(K, K.conj().T, atol=1e-12)
        assert np.linalg.eigvalsh(K).min() > 0
        np.testing.assert_allclose(K @ G, np.eye(3), atol=1e-12)


def test_cubic_symmetry():
    # A rotation by 90 degrees about a cube axis leaves the crystal invariant
    qx = np.array([1.0, 0.3, -0.7])
    qy = np.array([0.0, 0.8, 0.4])
    reference = AnisotropicGreensFunction(1.0, 0.5, 0.3).stiffness(qx, qy)
    rotated = AnisotropicGreensFunction(1.0, 0.5, 0.3, R=_rotation([1, 0, 0], np.pi / 2)).stiffness(qx, qy)
    np.testing.assert_allclose(rotated, reference, atol=1e-12)


def test_rotation_of_nearly_isotropic_crystal():
    qx = np.array([1.0, 0.3, -0.7])
    qy = np.array([0.0, 0.8, 0.4])
    C11, C44 = 1.0, 0.3
    anisotropy = 1e-3
    C12 = C11 - 2 * C44 + anisotropy
    reference = AnisotropicGreensFunction(C11, C12, C44).stiffness(qx, qy)
    rotated = AnisotropicGreensFunction(C11, C12, C44, R=_rotation([1, 2, 3], 0.7)).stiffness(qx, qy)
    assert np.abs(rotated - reference).max() < anisotropy * np.abs(reference).max()
