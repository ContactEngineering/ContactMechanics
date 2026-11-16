#
# Copyright 2016, 2020 Lars Pastewka
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
Greenwood-Tripp model for the contact of rough spheres
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk


# Note on notation in paper and scipy: K(x) = ellipk(x**2), E(x) = ellipe(x**2)


def _Fn(h, n, phi):
    v, err = quad(lambda s: (s - h) ** n * phi(s), h, np.inf)
    return v


def Fn(h, n, phi=lambda s: np.exp(-(s**2) / 2) / ((2 * np.pi) ** (1 / 2))):
    r"""
    Returns:
    --------
         :math:`\int_h^\infty (s-h)^n \phi(s) ds`
    """
    if np.isscalar(h):
        return _Fn(h, n, phi)
    else:
        return np.array([_Fn(_h, n, phi) for _h in h])


# Use asymptotic expressions for small and large values
def s(ξ, ξ1=1e-5, ξ2=1e5):
    r"""
    Returns:
    --------

    .. math ::

        2/\pi*E(ξ)+(ξ^2-1)*K(ξ) \text{     if  } ξ<=1

        2/\pi*ξ*E(1/ξ) \text{     if  } ξ>1

    Asymptotic cases are approximated by the asymptotic expressions.
    """

    def B(x):
        return ellipe(x**2) + (x**2 - 1) * ellipk(x**2)

    m1 = ξ < ξ1
    m4 = ξ > ξ2
    m = ξ < 1
    m2 = np.logical_and(m, np.logical_not(m1))
    m3 = np.logical_and(np.logical_not(m), np.logical_not(m4))
    r = np.zeros_like(ξ)
    r[m1] = ξ[m1] ** 2 / 2
    r[m4] = ξ[m4]
    r[m2] = 2 / np.pi * B(ξ[m2])
    r[m3] = 2 / np.pi * ξ[m3] * ellipe(1 / ξ[m3] ** 2)
    return r


_s = s


# Use asymptotic expressions for small and large values
def ξ(s, s1=1e-5, s2=1e5):
    """
    Inverse of `s`. Asymptotic cases are approximated by the asymptotic
    expressions.
    """
    # Note: The argument `s` shadows the function `s`, which is inverted here
    def ξnum(s0):
        return (
            brentq(lambda ξ: _s(np.array([ξ]))[0] - s0, 1e-12, 1e12)
            if np.isscalar(s0)
            else np.array([brentq(lambda ξ: _s(np.array([ξ]))[0] - _s0, 1e-12, 1e12) for _s0 in s0])
        )

    m1 = s < s1
    m3 = s > s2
    m2 = np.logical_and(np.logical_not(m1), np.logical_not(m3))
    r = np.zeros_like(s)
    r[m1] = np.sqrt(2 * s[m1])
    r[m3] = s[m3]
    r[m2] = ξnum(s[m2])
    return r


def _displacements(p, ρ, rhomax, eps):
    """
    Displacements due to the pressure p on the radial grid ρ.

    Returns
    -------
    w : np.ndarray
        Displacements.
    w0 : float
        Displacement at the center.
    """
    pint = interp1d(ρ, p)
    w0 = simpson(p, x=ρ)
    w = np.zeros_like(ρ)
    w[0] = w0
    for i, _ρ in enumerate(ρ[1:], start=1):
        ξ = np.linspace(0, (rhomax - eps) / _ρ, len(ρ))
        w[i] = simpson(_ρ * pint(_ρ * ξ), x=s(ξ))
    return w, w0


def GreenwoodTripp(d, μ, rhomax=5, n=100, eps=1e-6, tol=1e-6, mix=0.1, maxiter=1000):
    """
    Greenwood-Tripp solution for the contact of rough spheres.
    See: Greenwood, Tripp, J. Appl. Mech. 34, 153 (1967)
    Symbols correspond to notation in this paper.

    The pressure is computed self-consistently by a damped fixed-point
    iteration: p = μ F_{3/2}(d + ρ² + w(p) - w(p, 0)).

    Parameters
    ----------
    d : float
        Dimensionless displacement d* of the sphere, d* = d / σ where σ is the
        standard deviation of the height distribution (i.e. the rms height).
    μ : float
        Material parameter, μ=8/3 η σ sqrt(2 B β) where η is the density of
        asperities, B is the sphere radius and β the asperity radius.
    rhomax : float
        Radius cutoff (in units of sqrt(2 B σ), like the radial coordinate).
        Pressure is assumed to be zero beyond this radius.
    n : int
        Number of grid points.
    eps : float
        Small number for numerical regularization.
    tol : float
        Convergence tolerance: maximum change of the pressure (in units of
        E* sqrt(σ / 8B)) under one undamped fixed-point update.
    mix : float
        Mixing of solution between consecutive steps.
    maxiter : int
        Maximum number of iterations. A RuntimeError is raised if the
        iteration does not converge.

    Returns
    -------
    w : array
        Displacement in radial direction in units of σ.
    p : array
        Pressure in radial direction in units of E* sqrt(σ / 8B) where E* is
        the contact modulus.
    rho : array
          Radial coordinate in units of sqrt(2 B σ)
    """
    ρ = np.linspace(0, rhomax, n)
    w = np.zeros_like(ρ)
    w0 = 0
    p = np.zeros_like(ρ)
    for it in range(maxiter + 1):
        # Undamped update of the pressure for the current displacements
        p_new = μ * Fn(d + ρ**2 + w - w0, 3 / 2)
        # Note: Testing the change of the damped iterate would test
        # mix * residual, i.e. a tolerance of tol / mix
        if np.abs(p_new - p).max() <= tol:
            return w - w0, p_new, ρ
        p = mix * p_new + (1 - mix) * p
        w, w0 = _displacements(p, ρ, rhomax, eps)
    raise RuntimeError(
        "Maximum number of iterations (={}) " "exceeded.".format(maxiter)
    )
