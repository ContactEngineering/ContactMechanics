#
# Copyright 2017, 2020 Lars Pastewka
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
Westergaard solution for partial contact of a sinusoidal periodic
indenter.
See: H.M. Westergaard, Trans. ASME, J. Appl. Mech. 6, 49 (1939)
"""

from __future__ import division

import numpy as np


def _pressure(x, contact_radius=None, mean_pressure=None):
    """
    Compute pressure for Westergaard solution of unit amplitude.

    Parameters
    ----------
    x : array_like
        Fractional positions array
    contact_radius : float
        Dimensionless contact radius
    mean_pressure : float
        Dimensionless mean pressure

    Returns
    -------
    p : array
        Non-dimensional pressure
    """
    psi = np.pi * x
    if contact_radius is not None:
        psia = np.pi * contact_radius
        mean_pressure = np.pi * np.sin(psia) ** 2
    elif mean_pressure is not None:
        psia = np.arcsin(np.sqrt(mean_pressure / np.pi))
    p = np.zeros_like(x)
    m = np.cos(psi) > np.cos(psia)
    if m.sum() > 0:
        p[m] = 2 * np.cos(psi[m]) / (np.sin(psia) ** 2) * np.sqrt(
            np.sin(psia) ** 2 - np.sin(psi[m]) ** 2)
    return p
