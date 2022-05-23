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

r"""
Westergaard solution for partial contact of a sinusoidal periodic
indenter.
See: H.M. Westergaard, Trans. ASME, J. Appl. Mech. 6, 49 (1939)

The indenter geometry is

.. math ::

    2 h sin^2(\pi x / \lambda)

Nommenclature:
---------------

- :math:`E^*`: contact modulus
- :math:`\lambda`: period of the sinusoidal indenter
- :math:`h`: amplitude of the sinusoidal indenter. peak to tale distance
  is :math:`2h`
- :math:`a`: half contact width, "contact radius"


"""

import numpy as np
from numpy import sqrt, sin, cos, pi, log


def _map_x_on_0_05(x):
    return abs(np.mod(x + 0.5, 1.) - 0.5)


def _pressure(x, contact_radius=None, mean_pressure=None):
    """
    Compute pressure for Westergaard solution of unit amplitude.

    Parameters
    ----------
    x : array_like
        Dimensionless positions array
    contact_radius : float
        Dimensionless contact radius
    mean_pressure : float
        mean pressure nondimensionalised by pi * Es * h0 / lam

    Returns
    -------
    p : array
        Non-dimensional pressure


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-2.,2., 500)
    >>> l,= ax.plot(x, _pressure(x, 0.1), )
    >>> leg = ax.legend()
    >>> plt.show(block=True)
    """
    x = _map_x_on_0_05(x)

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


def displacements(x, a):
    """
    Displacements u in units of 2h, the geometry is
    :math:`2h sin^2(pi x / lambda)`

    Parameters
    ----------
    x: float or np.array
        distance from the indenter tip contact point in units of lambda
    a: float
        half contact length in units of lambda

    Returns
    -------
    Displacents u in units of 2h

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(-1.,1., 500)
    >>> l,= ax.plot(x, displacements(x, 0.1), label="displacements")
    >>> l2, = ax.plot(x, -np.sin(np.pi * x)**2, "--k", label="geometry")
    >>> ax.invert_yaxis()
    >>> leg = ax.legend()
    >>> plt.show(block=True)

    """

    returnscalar = not hasattr(x, "__iter__")

    x = _map_x_on_0_05(x)

    if returnscalar:
        x = np.asarray(x)

    sl_out = np.logical_and(x > a, x < 1 - a)

    chi1 = np.zeros_like(x)
    chi2 = np.zeros_like(x)

    chi1[sl_out] = sqrt(sin(pi * x[sl_out]) ** 2 - sin(pi * a) ** 2)
    chi2[sl_out] = log((sin(pi * x[sl_out]) + chi1[sl_out]) / sin(pi * a))

    res = - sin(pi * x) ** 2 + chi1 * sin(pi * x) - sin(pi * a) ** 2 * chi2

    if returnscalar:
        return res.item()
    else:
        return res


def gap(x, a):
    """
    gap g in units of 2h, the geometry is :math:`2h sin^2(pi x / lambda)`

    Parameters
    ----------
    x: float or np.array
        distance from the first contact point in units of lambda
    a: float
        half contact length in units of lambda

    Returns
    -------
    Displacents u in units of 2h
    """
    return sin(pi * x) ** 2 + displacements(x, a)


def mean_displacement(a):
    r"""
    mean gap would be 1 / 2 + mean displacement

    Parameters
    ----------
    a : float or np.array
        half contactwidth in units of the wavelength lambda

    Returns
    -------
    mean displacement in units of 2 h

    Examples
    ---------
    >>> import matplotlib.pyplot as plt
    >>> a = np.linspace(0,0.5)
    >>> l,= plt.plot(a, mean_displacement(a))
    >>> plt.show()
    """
    return -1 / 2 + 1 / 2 * cos(pi * a) ** 2 + sin(pi * a) ** 2 * log(
        sin(pi * a))


def elastic_energy(mean_pressure):
    r"""
    elastic energy along the equilibrium curve

    formula for the JKR contact but with  alpha = 0


    Parameters
    ----------
    mean_pressure: float or np.array
        mean pressure in units of :math:`\pi Es h/ lambda`

    Returns
    -------
    energy per unit area in units of :math:`h p_{wfc}`

    with :math:`p_{wfc} = \pi E^* h/\lambda`
    """

    assert np.all(mean_pressure <= 1)
    return (1 / 4 - log(mean_pressure) / 2) * mean_pressure ** 2


def elastic_energy_a(a):
    r"""
    elastic energy in dependence of a
    along the equilibrium curve


    Parameters
    ----------
    a: float or np.array
        half contact width in units of the wavelength :math:`\lambda`

    Returns
    -------
    energy per unit area in units of :math:`h p_{wfc}`

    with :math:`p_{wfc} = pi E^* h/\lambda`
    """
    return (1 / 4 - log(sin(pi * a) ** 2) / 2) * sin(pi * a) ** 4


def contact_radius(mean_pressure):
    r"""

    Johnson, K. L.
    International Journal of Solids and Structures 32, 423–430 (1995)

    Equation (4)

    Parameters
    ----------
    mean_pressure: float, array_like
        mean pressure in units of :math:`\pi E^* h / \lambda`
    Returns
    -------
    contact radius in units of :math:`\lambda`

    """
    return 1 / np.pi * np.arcsin(np.sqrt(mean_pressure))


def mean_pressure(contact_radius):
    r"""

    Johnson, K. L.
    International Journal of Solids and Structures 32, 423–430 (1995)

    Equation (4)


    Parameters
    ----------
    contact_radius: float
     in units of :math:`\lambda`

    Returns
    -------
    mean pressure: float, array_like
            mean pressure in units of :math:`\pi E^* h / \lambda`

    """

    return np.sin(np.pi * contact_radius)**2
