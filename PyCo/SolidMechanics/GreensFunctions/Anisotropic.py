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

#
# Anisotropic continuum surface Green's function
#

import numpy as np
from scipy.linalg import null_space


###

class AnisotropicGreensFunction(object):
    def __init__(self, C11, C12, C44):
        self._C11 = C11
        self._C12 = C12
        self._C44 = C44
        self._C = np.array([[C11, C12, C12, 0, 0, 0],
                            [C12, C11, C12, 0, 0, 0],
                            [C12, C12, C11, 0, 0, 0],
                            [0, 0, 0, C44, 0, 0],
                            [0, 0, 0, 0, C44, 0],
                            [0, 0, 0, 0, 0, C44]])

    def elasticity_tensor(self, i, j, k, l):
        Voigt_ij = i
        if i != j:
            Voigt_ij = 6 - i - j
        Voigt_kl = k
        if k != l:
            Voigt_kl = 6 - k - l
        return self._C[Voigt_ij, Voigt_kl]

    def bulkop(self, qx, qy, qz):
        """
        Return the linear operator M_il = -C_ijkl q_j q_k

        Arguments
        ---------
        q : 3-vector
            Components of the wavevector

        Returns
        -------
        M : 3x3-matrix
            Linear operator
        """
        q = (qx, qy, qz)
        M = np.zeros((3, 3), dtype=complex)
        for i, j, k, l in np.ndindex(3, 3, 3, 3):
            M[i, l] += -self.elasticity_tensor(i, j, k, l) * q[j] * q[k]
        return M

    def find_eigenvalues(self, qx, qy):
        # We know that det(M) has the form c0 + c2*Q^2 + c4*Q^4 + c6*Q^6, but we don't have the prefactors explicitly.
        # We first need to construct them here by evaluating:
        # Q = 0: det(M) = c0
        # Q = 1: det(M) = c0 + c2 + c4 + c6
        # Q = 2: det(M) = c0 + 4*c2 + 16*c4 + 64*c6
        # Q = 3: det(M) = c0 + 9*c2 + 81*c4 + 729*c6

        fac1 = 1
        fac2 = 2
        fac3 = 3
        A = np.array([[0, 0, 0, 1],
                      [fac1 ** 6, fac1 ** 4, fac1 ** 2, 1],
                      [fac2 ** 6, fac2 ** 4, fac2 ** 2, 1],
                      [fac3 ** 6, fac3 ** 4, fac3 ** 2, 1]])
        b = np.array([np.linalg.det(self.bulkop(qx, qy, 0)),
                      np.linalg.det(self.bulkop(qx, qy, 1j * fac1)),
                      np.linalg.det(self.bulkop(qx, qy, 1j * fac2)),
                      np.linalg.det(self.bulkop(qx, qy, 1j * fac3))])
        p = np.linalg.solve(A, b)
        r = np.roots(p)

        # We need to take the sqrt because we have the roots of the equation c0 + c2*Q + c4*Q^2 + C6*Q^3
        return np.sqrt(r)

    def find_eigenvectors(self, qx, qy, qz, rcond=1e-6):
        eta = []
        for _qz in qz:
            M = self.bulkop(qx, qy, _qz)
            _eta = null_space(M, rcond=rcond)
            if _eta.shape[1] != 1:
                raise RuntimeError(f'Null space for wavevector {qx},{qy},{_qz} spanned by {_eta.shape[1]} vectors, '
                                   'but should be spanned by a single one.')
            eta += [_eta[:, 0]]
        return eta

    def make_U(selfself, qz, eta, z=None):
        U = np.zeros((3, len(qz)), dtype=complex)
        if z is None:
            for k, alpha in np.ndindex(3, len(qz)):
                U[k, alpha] = eta[alpha][k]
        else:
            for k, alpha in np.ndindex(3, len(qz)):
                U[k, alpha] = eta[alpha][k] * np.exp(1j * qz[alpha] * z)
        return U

    def make_F(self, qx, qy, qz, eta):
        q = [(qx, qy, _qz) for _qz in qz]
        F = np.zeros((3, len(qz)), dtype=complex)
        for i, k, alpha, l in np.ndindex(3, 3, len(qz), 3):
            F[i, alpha] += 1j * self.elasticity_tensor(i, 2, k, l) * q[alpha][k] * eta[alpha][l]
        return F

    def make_U_and_F(self, qx, qy):
        qz = -1j * self.find_eigenvalues(qx, qy)
        eta = self.find_eigenvectors(qx, qy, qz)
        return self.make_U(qz, eta), self.make_F(qx, qy, qz, eta)

    def _greens_function(self, qx, qy):
        U, F = self.make_U_and_F(qx, qy)
        return np.linalg.solve(F.T, U.T).T

    def _stiffness(self, qx, qy):
        if abs(qx) < 1e-6 and abs(qy) < 1e-6:
            return np.zeros((3, 3))
        U, F = self.make_U_and_F(qx, qy)
        return np.linalg.solve(U.T, F.T).T

    def greens_function(self, qx, qy):
        if np.isscalar(qx) and np.isscalar(qy):
            return self._greens_function(qx, qy)

        gf = []
        for _qx, _qy in zip(qx, qy):
            gf += [self._greens_function(_qx, _qy)]
        return np.array(gf)

    def stiffness(self, qx, qy):
        if np.isscalar(qx) and np.isscalar(qy):
            return self._stiffness(qx, qy)

        gf = []
        for _qx, _qy in zip(qx, qy):
            gf += [self._stiffness(_qx, _qy)]
        return np.array(gf)
