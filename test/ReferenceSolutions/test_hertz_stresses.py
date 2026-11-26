"""
Consistency tests of the stress fields of the Hertz solution.
"""

import numpy as np
import pytest

import ContactMechanics.ReferenceSolutions.Hertz as Hertz


@pytest.fixture
def points():
    x = np.array([0.3, 0.8, 1.5, 0.5])
    z = np.array([0.2, 0.5, 0.3, 1e-3])
    return x, z


@pytest.mark.parametrize('poisson', [0.3, 0.5])
def test_cartesian_and_cylindrical_stress(points, poisson):
    # On the x-axis, the Cartesian stress is the cylindrical stress
    x, z = points
    zeros = np.zeros_like(x)
    sxx, syy, szz, syz, sxz, sxy = Hertz.stress_Cartesian(x, zeros, z, poisson)
    stt, srr, szz_cylindrical, srz = Hertz.stress(x, z, poisson)
    np.testing.assert_allclose(sxx, srr, atol=1e-12)
    np.testing.assert_allclose(syy, stt, atol=1e-12)
    np.testing.assert_allclose(szz, szz_cylindrical, atol=1e-12)
    np.testing.assert_allclose(sxz, srz, atol=1e-12)
    np.testing.assert_allclose(syz, 0, atol=1e-12)
    np.testing.assert_allclose(sxy, 0, atol=1e-12)

    # Rotational symmetry: on the y-axis, x and y are swapped
    sxx, syy, szz, syz, sxz, sxy = Hertz.stress_Cartesian(zeros, x, z, poisson)
    np.testing.assert_allclose(syy, srr, atol=1e-12)
    np.testing.assert_allclose(sxx, stt, atol=1e-12)
    np.testing.assert_allclose(syz, srz, atol=1e-12)


def test_centerline_stress():
    poisson = 0.3
    z = np.array([0.1, 0.5, 1.0, 2.0])
    srr, szz = Hertz.centerline_stress(z, poisson)
    # Interior stress on the axis
    stt, srr_interior, szz_interior, srz = Hertz.stress(np.zeros_like(z), z, poisson)
    np.testing.assert_allclose(srr, srr_interior, atol=1e-12)
    np.testing.assert_allclose(szz, szz_interior, atol=1e-12)
    # Cartesian stress on the axis
    sxx, syy, szz_cartesian, syz, sxz, sxy = Hertz.stress_Cartesian(np.zeros_like(z), np.zeros_like(z), z, poisson)
    np.testing.assert_allclose(sxx, srr, atol=1e-12)
    np.testing.assert_allclose(syy, srr, atol=1e-12)
    np.testing.assert_allclose(szz_cartesian, szz, atol=1e-12)
    # Normal stress at the surface equals the maximum pressure
    np.testing.assert_allclose(Hertz.centerline_stress(np.array([1e-12]), poisson)[1], -1)


def test_surface_displacements():
    r = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
    displacements = Hertz.surface_displacements(r)
    inside = r < 1
    np.testing.assert_allclose(displacements[inside], -np.pi * (2 - r[inside] ** 2) / 4)
    # Continuous at the contact edge
    np.testing.assert_allclose(Hertz.surface_displacements(np.array([1 - 1e-12, 1 + 1e-12])), -np.pi / 4, rtol=1e-5)
    # Displacement decays outside of the contact
    assert np.all(np.diff(displacements[r >= 1]) > 0)


def test_tangential_loading():
    poisson = 0.3
    z = np.array([0.1, 0.5, 1.0])
    x = np.full_like(z, 0.4)
    y = np.full_like(z, 0.3)
    stress = Hertz.stress_for_tangential_loading(x, y, z, poisson)
    # Mirroring x -> -x reverses the tangential load in x-direction, hence
    # s_ij(-x) = -p_i p_j s_ij(x) with p = (-1, 1, 1): sxx, syy, szz and syz
    # are odd in x, sxz and sxy are even. Mirroring y -> -y leaves the load
    # unchanged: syz and sxy are odd in y, all other components even.
    mirrored_x = Hertz.stress_for_tangential_loading(-x, y, z, poisson)
    for component, sign in zip(range(6), [-1, -1, -1, -1, 1, 1]):
        np.testing.assert_allclose(mirrored_x[component], sign * stress[component], atol=1e-12)
    mirrored_y = Hertz.stress_for_tangential_loading(x, -y, z, poisson)
    for component, sign in zip(range(6), [1, 1, 1, -1, 1, -1]):
        np.testing.assert_allclose(mirrored_y[component], sign * stress[component], atol=1e-12)
    # The value on the axis is the limit of the off-axis expressions
    on_axis = Hertz.stress_for_tangential_loading(np.zeros_like(z), np.zeros_like(z), z, poisson)
    close_to_axis = Hertz.stress_for_tangential_loading(np.full_like(z, 1e-3), np.zeros_like(z), z, poisson)
    for a, b in zip(on_axis, close_to_axis):
        np.testing.assert_allclose(a, b, atol=1e-2)


def test_hertz_relations():
    contact_modulus, radius, load = 2.0, 3.0, 5.0
    penetration = Hertz.penetration(load, radius, contact_modulus)
    contact_radius, max_pressure = Hertz.radius_and_pressure(load, radius, contact_modulus)
    np.testing.assert_allclose(contact_radius ** 2, radius * penetration)
    np.testing.assert_allclose(max_pressure, 3 * load / (2 * np.pi * contact_radius ** 2))
    np.testing.assert_allclose(Hertz.max_pressure__penetration(penetration, radius, contact_modulus), max_pressure)
