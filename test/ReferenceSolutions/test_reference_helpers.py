"""
Tests of the helper functions of the Westergaard and Greenwood-Tripp
reference solutions.
"""

import numpy as np
import pytest

import ContactMechanics.ReferenceSolutions.GreenwoodTripp as GT
import ContactMechanics.ReferenceSolutions.Westergaard as Westergaard


def test_westergaard_pressure():
    x = np.linspace(-1, 1, 2001)
    a = 0.2
    pressure = Westergaard._pressure(x, contact_radius=a)
    np.testing.assert_allclose(Westergaard._pressure(x, mean_pressure=np.pi * np.sin(np.pi * a) ** 2), pressure)
    # Periodic, no pressure outside of the contact
    np.testing.assert_allclose(Westergaard._pressure(x + 1, contact_radius=a), pressure, atol=1e-10)
    assert np.all(pressure[np.abs(x - np.round(x)) > a] == 0)


def test_westergaard_displacements_and_gap():
    a = 0.2
    x = np.linspace(0, 1, 1001)
    gap = Westergaard.gap(x, a)
    # No gap inside the contact, positive gap outside
    np.testing.assert_allclose(gap[(x < a) | (x > 1 - a)], 0, atol=1e-12)
    assert np.all(gap[(x > a + 1e-3) & (x < 1 - a - 1e-3)] > 0)
    # Scalar input
    assert np.isscalar(Westergaard.displacements(0.5, a))
    np.testing.assert_allclose(Westergaard.displacements(0.5, a), Westergaard.displacements(np.array([0.5]), a)[0])
    # Mean displacement
    np.testing.assert_allclose(np.trapezoid(Westergaard.displacements(x, a), x=x), Westergaard.mean_displacement(a),
                               atol=1e-5)


def test_westergaard_energy():
    a = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(Westergaard.elastic_energy(Westergaard.mean_pressure(a)),
                               Westergaard.elastic_energy_a(a))


def test_greenwood_tripp_helpers():
    # Scalar and vector input of Fn
    np.testing.assert_allclose(GT.Fn(0.5, 1.5), GT.Fn(np.array([0.5]), 1.5)[0])
    # ξ is the inverse of s, including the asymptotic regimes
    ξ = np.array([1e-7, 1e-3, 0.5, 1.0, 2.0, 1e3, 1e7])
    np.testing.assert_allclose(GT.ξ(GT.s(ξ)), ξ, rtol=1e-6)
    with pytest.raises(RuntimeError):
        GT.GreenwoodTripp(1.0, 24.0425586841, maxiter=1)
