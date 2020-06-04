#
# Copyright 2019-2020 Lars Pastewka
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
# Isotropic continuum surface Green's function
#

import numpy as np


###

class IsotropicGreensFunction(object):
    def __init__(self, shear_modulus, Poisson_ratio):
        """
        Compute the surface Green's function for a linear elastic half-space
        with isotropic elastic constants.

        Parameters
        ----------
        shear_modulus : float
            Shear modulus
        Poisson_ratio : float
            Poisson ratio
        """
        self._mu = shear_modulus
        self._nu = Poisson_ratio

    def _greens_function(self, qx, qy):
        q = np.sqrt(qx ** 2 + qy ** 2)
        nu = self._nu
        G = np.array([[1 / q - nu * qx ** 2 / q ** 3,
                       -nu * qx * qy / q ** 3,
                       1j * (1 - 2 * nu) * qx / (2 * q ** 2)],
                      [-nu * qx * qy / q ** 3,
                       1 / q - nu * qy ** 2 / q ** 3,
                       1j * (1 - 2 * nu) * qy / (2 * q ** 2)],
                      [-1j * (1 - 2 * nu) * qx / (2 * q ** 2),
                       -1j * (1 - 2 * nu) * qy / (2 * q ** 2),
                       (1 - nu) / q]])
        return G / self._mu

    def _stiffness(self, qx, qy):
        if abs(qx) < 1e-6 and abs(qy) < 1e-6:
            return np.zeros((3, 3))
        return np.linalg.inv(self._greens_function(qx, qy))

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

    __call__ = stiffness
