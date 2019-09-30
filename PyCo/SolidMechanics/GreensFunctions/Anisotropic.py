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
from scipy.optimize import root_scalar


###

class AnisotropicGF(object):
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

    def bulkop(self, q):
        """
        Return the linear operator M_il = -C_ijkl q_j q_k

        Arguments
        ---------
        q : 3-vector
            Components of the wavevector

        Returns
        -------
        M_il : 3x3-matrix
            Linear operator
        """
        M_il = np.zeros((3, 3), dtype=complex)
        for i, j, k, l in np.ndindex(3, 3, 3, 3):
            Voigt_ij = i
            if i != j:
                Voigt_ij = 6 - i - j
            Voigt_kl = k
            if k != l:
                Voigt_kl = 6 - k - l
            C_ijkl = self._C[Voigt_ij, Voigt_kl]
            M_il[i, l] += -C_ijkl * q[j] * q[k]
        return M_il

    def find_qz(self, qx, qy):
        sol = root_scalar(lambda qz: np.linalg.det(self.bulkop([qx, qy, 1j * qz])),
                          bracket=(0, 5 * max(abs(qx), abs(qy))))
        return sol.root
