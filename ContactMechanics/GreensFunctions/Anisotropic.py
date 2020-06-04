#
# Copyright 2019-2020 Lars Pastewka
#           2019 wnoehring@simnetpc68.imtek.privat
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

#
# Anisotropic continuum surface Green's function
#

import numpy as np
from scipy.linalg import null_space


class AnisotropicGreensFunction(object):
    def __init__(self, C11, C12, C44, thickness=None, R=np.eye(3)):
        """
        Compute the surface Green's function for a linear elastic half-space
        with anisotropic elastic constants. The class supports generic
        elastic tensors but presently only cubic elastic constants can be
        passed to the constructor.

        Note that this class fails for an isotropic substrate. Use
        `IsotropicGreensFunction` instead for isotropic materials.

        Parameters
        ----------
        C11 : float
            C11 elastic constant
        C12 : float
            C12 elastic constant
        C44 : float
            C44 elastic constant (shear modulus)
        thickness : float
            Thickness of the elastic substrate. If None (default) then a
            substrate of infinite thickness will be computed.
        R : np.ndarray
            3x3 rotation matrix for rotation of the elastic constants.
        """
        self._C11 = C11
        self._C12 = C12
        self._C44 = C44
        self._C = np.array([[C11, C12, C12, 0, 0, 0],  # xx
                            [C12, C11, C12, 0, 0, 0],  # yy
                            [C12, C12, C11, 0, 0, 0],  # zz
                            [0, 0, 0, C44, 0, 0],  # yz
                            [0, 0, 0, 0, C44, 0],  # xz
                            [0, 0, 0, 0, 0, C44]])  # xy
        self._thickness = thickness
        det_R = np.linalg.det(R)
        if not np.isclose(det_R, 1.0):
            raise ValueError(
                "R is not a proper rotation matrix, det(R)={}".format(det_R))
        self._R = R
        C_tensor = np.zeros((3, 3, 3, 3))
        for i, j, k, l in np.ndindex(3, 3, 3, 3):
            C_tensor[i, j, k, l] = self.elasticity_tensor(i, j, k, l)
        C_tensor = np.einsum(
            'ig,jh,ghmn,km,ln', self._R, self._R, C_tensor, self._R, self._R
        )
        for i, j in np.ndindex(6, 6):
            self._C[i, j] = self.voigt_from_tensor(C_tensor, i, j)

    def elasticity_tensor(self, i, j, k, L):
        Voigt_ij = i
        if i != j:
            Voigt_ij = 6 - i - j
        Voigt_kl = k
        if k != L:
            Voigt_kl = 6 - k - L
        return self._C[Voigt_ij, Voigt_kl]

    def voigt_from_tensor(self, C_tensor, Voigt_ij, Voigt_kl):
        tensor_ij_for_voigt_ij = {
            0: (0, 0),
            1: (1, 1),
            2: (2, 2),
            3: (1, 2),
            4: (0, 2),
            5: (0, 1),
        }
        i, j = tensor_ij_for_voigt_ij[Voigt_ij]
        k, L = tensor_ij_for_voigt_ij[Voigt_kl]
        return C_tensor[i, j, k, L]

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
        # We know that det(M) has the form c0 + c2*Q^2 + c4*Q^4 + c6*Q^6, but
        # we don't have the prefactors explicitly. We first need to construct
        # them here by evaluating:
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

        # We need to take the sqrt because we have the roots of the equation
        # c0 + c2*Q + c4*Q^2 + C6*Q^3
        return np.sqrt(r)

    def find_eigenvectors(self, qx, qy, qz, rcond=1e-6):
        eta = []
        for _qz in qz:
            M = self.bulkop(qx, qy, _qz)
            _eta = null_space(M, rcond=rcond)
            if _eta.shape[1] != 1:
                raise RuntimeError(
                    'Null space for wavevector {},{},{} spanned by {} '
                    'vectors, but should be spanned by a single one.'
                    .format(qx, qy, _qz, _eta.shape[1]))
            eta += [_eta[:, 0]]
        return eta

    def _make_U(self, qz, eta, z=None):
        U = np.zeros((3, len(qz)), dtype=complex)
        if z is None:
            for k, alpha in np.ndindex(3, len(qz)):
                U[k, alpha] = eta[alpha][k]
        else:
            for k, alpha in np.ndindex(3, len(qz)):
                U[k, alpha] = eta[alpha][k] * np.exp(1j * qz[alpha] * z)
        return U

    def _make_F(self, qx, qy, qz, eta, thickness):
        q = [(qx, qy, _qz) for _qz in qz]
        F = np.zeros((len(qz), len(qz)), dtype=complex)
        # Traction boundary conditions on top
        for i, k, alpha, l in np.ndindex(3, 3, len(qz), 3):
            F[i, alpha] += 1j * self.elasticity_tensor(i, 2, k, l) * \
                           q[alpha][k] * eta[alpha][l]
        # Displacement boundary conditions on bottom
        if len(qz) > 3:
            for i, alpha in np.ndindex(3, len(qz)):
                F[i + 3, alpha] = np.exp(-1j * q[alpha][2] * thickness) * \
                                  eta[alpha][i]
        return F

    def _make_U_and_F(self, qx, qy, thickness, exp_tol=100):
        _qz = self.find_eigenvalues(qx, qy)
        # If thickness*qz > some threshold, then we need to solve for the
        # problem of infinite thickness, otherwise we get floating point
        # issues when evaluating exp(-thickness*qz) in _make_F
        if thickness is None or np.max(np.real(thickness * _qz)) > exp_tol:
            qz = -1j * _qz
        else:
            qz = np.append(-1j * _qz, 1j * _qz)
        eta = self.find_eigenvectors(qx, qy, qz)
        return self._make_U(qz, eta), self._make_F(qx, qy, qz, eta, thickness)

    def _gamma_stiffness(self):
        """Returns the 3x3 stiffness matrix at the Gamma point (q=0)"""
        # Voigt components: xz = 4, yz = 3, zz = 2
        return np.array([[self._C[4, 4], self._C[3, 4], self._C[2, 4]],
                         [self._C[3, 4], self._C[3, 3], self._C[2, 3]],
                         [self._C[2, 4], self._C[2, 3],
                          self._C[2, 2]]]) / self._thickness

    def _greens_function(self, qx, qy, thickness, zero_tol=1e-6):
        if thickness is not None and abs(qx) < zero_tol and abs(qy) < zero_tol:
            # This is zero wavevector. We use the analytical solution in this
            # case.
            return np.linalg.inv(self._gamma_stiffness())
        U, F = self._make_U_and_F(qx, qy, thickness)
        return np.linalg.solve(F.T, U.T)[:3, :]

    def _stiffness(self, qx, qy, thickness, zero_tol=1e-6):
        if abs(qx) < zero_tol and abs(qy) < zero_tol:
            if thickness is None:
                return np.zeros((3, 3))
            else:
                return self._gamma_stiffness()
        if thickness is None:
            U, F = self._make_U_and_F(qx, qy, thickness)
            return np.linalg.solve(U.T, F.T)
        else:
            return np.linalg.inv(
                self._greens_function(qx, qy, thickness, zero_tol=zero_tol))

    def greens_function(self, qx, qy, zero_tol=1e-6):
        # Note: Normalization of the q vectors is required for numerical
        # stability
        abs_q = np.sqrt(qx ** 2 + qy ** 2)
        abs_q[abs_q == 0] = 1
        thickness = [None] * len(
            abs_q) if self._thickness is None else self._thickness * abs_q

        if np.isscalar(qx) and np.isscalar(qy):
            return self._greens_function(qx / abs_q, qy / abs_q, thickness,
                                         zero_tol=zero_tol) / abs_q

        gf = []
        for _qx, _qy, _abs_q, _thickness in zip(qx, qy, abs_q, thickness):
            gf += [
                self._greens_function(_qx / _abs_q, _qy / _abs_q, _thickness,
                                      zero_tol=zero_tol) / _abs_q]
        return np.array(gf)

    def stiffness(self, qx, qy, zero_tol=1e-6):
        # Note: Normalization of the q vectors is required for numerical
        # stability
        abs_q = np.sqrt(qx ** 2 + qy ** 2)
        abs_q[abs_q == 0] = 1
        thickness = [None] * len(
            abs_q) if self._thickness is None else self._thickness * abs_q

        if np.isscalar(qx) and np.isscalar(qy):
            return self._stiffness(qx / abs_q, qy / abs_q, thickness,
                                   zero_tol=zero_tol) * abs_q

        gf = []
        for _qx, _qy, _abs_q, _thickness in zip(qx, qy, abs_q, thickness):
            gf += [self._stiffness(_qx / _abs_q, _qy / _abs_q, _thickness,
                                   zero_tol=zero_tol) * _abs_q]
        return np.array(gf)

    __call__ = stiffness
