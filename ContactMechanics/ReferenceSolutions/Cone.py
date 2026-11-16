#
# Copyright 2020 Lars Pastewka
#           2020 Antoine Sanner
#           2019 Lintao Fang
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
Indentation of an elastic half-space by a rigid cone.

Reference:
K.L. Johnson, Contact Mechanics, Cambridge University Press (1985),
Chapter 5, p. 112.

Nomenclature:
- alpha: semi-angle of the cone
- beta = pi / 2 - alpha: angle between the surface of the cone and the
  undeformed surface of the half-space
- D: rigid body penetration
- a: contact radius
- E*: contact modulus
"""

import numpy as np


def load_and_mean_pressure(alpha):
    """
    Parameters
    ----------
    alpha : float
        Semi-angle of the cone

    Returns
    -------
    load : float
        External force / (contact modulus * contact area), F / (E* pi a^2)
    mean_pressure : float
        Mean pressure / contact modulus, p_m / E*
    """
    beta = np.pi / 2 - alpha
    mean_pressure = np.tan(beta) / 2
    return mean_pressure, mean_pressure


def contact_radius_and_area(alpha):
    """
    Parameters
    ----------
    alpha : float
        Semi-angle of the cone

    Returns
    -------
    contact_radius : float
        Contact radius / penetration, a / D
    contact_area : float
        Contact area / penetration^2, pi a^2 / D^2
    """
    beta = np.pi / 2 - alpha
    contact_radius = 2 / (np.pi * np.tan(beta))
    return contact_radius, np.pi * contact_radius ** 2


def deformation(r, alpha):
    """
    Normal displacement of the surface of the half-space.

    Parameters
    ----------
    r : array_like
        Radial distance from the tip of the cone / penetration, r / D
    alpha : float
        Semi-angle of the cone

    Returns
    -------
    deformation : np.ndarray
        Normal displacement / penetration, u / D. Inside the contact, the
        surface follows the cone, u = D - r tan(beta).
    """
    r = np.asarray(r, dtype=float)
    beta = np.pi / 2 - alpha
    contact_radius, _ = contact_radius_and_area(alpha)

    deformation = np.empty_like(r)
    inside = r <= contact_radius
    deformation[inside] = 1 - r[inside] * np.tan(beta)
    outside = np.logical_not(inside)
    ratio = r[outside] / contact_radius
    deformation[outside] = 2 * (np.arcsin(1 / ratio) - ratio + np.sqrt(ratio ** 2 - 1)) / np.pi
    return deformation


def pressure(r, alpha):
    """
    Contact pressure. The pressure diverges logarithmically at the tip of the
    cone.

    Parameters
    ----------
    r : array_like
        Radial distance from the tip of the cone / penetration, r / D
    alpha : float
        Semi-angle of the cone

    Returns
    -------
    pressure : np.ndarray
        Pressure / mean pressure, p / p_m
    """
    r = np.asarray(r, dtype=float)
    contact_radius, _ = contact_radius_and_area(alpha)
    pressure = np.zeros_like(r)
    inside = r < contact_radius
    with np.errstate(divide='ignore'):
        pressure[inside] = np.arccosh(contact_radius / r[inside])
    return pressure
