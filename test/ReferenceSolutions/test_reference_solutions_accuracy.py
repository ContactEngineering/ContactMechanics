"""
Tests of the accuracy and consistency of the reference solutions.
"""

import numpy as np
import pytest
from SurfaceTopography import Topography

import ContactMechanics.ReferenceSolutions.Cone as Cone
import ContactMechanics.ReferenceSolutions.GreenwoodTripp as GT
import ContactMechanics.ReferenceSolutions.Hertz as Hertz
from ContactMechanics import FreeFFTElasticHalfSpace, make_system


def test_cone_contact_area():
    alpha = np.radians(70)
    contact_radius, contact_area = Cone.contact_radius_and_area(alpha)
    np.testing.assert_allclose(contact_area, np.pi * contact_radius ** 2)


def test_cone_deformation():
    alpha = np.radians(80)
    contact_radius, _ = Cone.contact_radius_and_area(alpha)
    r = np.linspace(0, 3 * contact_radius, 301)
    deformation = Cone.deformation(r, alpha)
    # Independent of the other sampling points
    np.testing.assert_allclose(Cone.deformation(r[[10]], alpha), deformation[10])
    np.testing.assert_allclose(Cone.deformation(r[[150]], alpha), deformation[150])
    # Rigid body penetration at the tip, continuous at the contact edge
    np.testing.assert_allclose(deformation[0], 1)
    np.testing.assert_allclose(Cone.deformation([contact_radius * (1 + 1e-9)], alpha),
                               Cone.deformation([contact_radius], alpha), atol=1e-6)
    # Surface follows the cone inside the contact
    inside = r <= contact_radius
    np.testing.assert_allclose(deformation[inside], 1 - r[inside] * np.tan(np.pi / 2 - alpha))


def test_cone_numerical():
    # Compare with a numerical calculation
    nb_grid_pts = 128
    tan_beta = 0.1
    alpha = np.pi / 2 - np.arctan(tan_beta)
    contact_modulus = 2.0
    penetration = 0.03

    x = (np.arange(nb_grid_pts) - nb_grid_pts // 2) / nb_grid_pts
    r = np.sqrt(x.reshape(-1, 1) ** 2 + x.reshape(1, -1) ** 2)
    system = make_system(substrate=FreeFFTElasticHalfSpace((nb_grid_pts, nb_grid_pts), contact_modulus, (1., 1.)),
                         surface=Topography(-r * tan_beta, (1., 1.)))
    result = system.minimize_proxy(offset=penetration, pentol=1e-10)
    assert result.success

    contact_radius, contact_area = Cone.contact_radius_and_area(alpha)
    load, mean_pressure = Cone.load_and_mean_pressure(alpha)
    np.testing.assert_allclose(result.jac.sum(), load * contact_modulus * contact_area * penetration ** 2, rtol=1e-3)
    np.testing.assert_allclose((result.jac > 0).mean(), contact_area * penetration ** 2, rtol=0.02)
    np.testing.assert_allclose(result.x[:nb_grid_pts, :nb_grid_pts],
                               Cone.deformation(r / penetration, alpha) * penetration, atol=1e-2 * penetration)


def test_cone_pressure():
    alpha = np.radians(70)
    contact_radius, _ = Cone.contact_radius_and_area(alpha)
    r = np.linspace(1e-3, 2 * contact_radius, 2001)
    pressure = Cone.pressure(r, alpha)
    assert np.all(pressure[r >= contact_radius] == 0)
    # The mean pressure is unity
    inside = r < contact_radius
    mean_pressure = np.trapezoid(2 * np.pi * r[inside] * pressure[inside], x=r[inside]) / (np.pi * contact_radius ** 2)
    np.testing.assert_allclose(mean_pressure, 1, rtol=1e-2)


def test_hertz_surface_stress_near_center():
    # The radial and azimuthal stresses deviate quadratically from the value
    # at the center; the previous implementation cancelled for small r
    poisson = 0.3
    r = np.array([0, 1e-7, 2e-6, 1e-4])
    pz, sr, stheta = Hertz.surface_stress(r, poisson)
    center = -(1 + 2 * poisson) / 2
    np.testing.assert_allclose(sr[0], center)
    np.testing.assert_allclose(stheta[0], center)
    assert np.all(np.abs(sr - center) <= r ** 2)
    assert np.all(np.abs(stheta - center) <= r ** 2)


def test_hertz_stress_at_surface():
    # The interior stress approaches the surface stress
    poisson = 0.3
    r = np.array([0, 1e-7, 0.3, 0.7, 0.95, 1.5])
    z = np.full_like(r, 1e-12)
    stt, srr, szz, srz = Hertz.stress(r, z, poisson)
    pz, sr, stheta = Hertz.surface_stress(r, poisson)
    np.testing.assert_allclose(srr, sr, atol=1e-10)
    np.testing.assert_allclose(stt, stheta, atol=1e-10)
    np.testing.assert_allclose(szz, -pz, atol=1e-10)


@pytest.mark.parametrize('r, z, reference', [
    # Reference values computed with 50-digit arithmetic (mpmath) from the
    # same expressions: stt, srr, szz, srz
    (0.5, 1e-8, (-0.706538393669869, -0.6791022115445288, -0.8660254037844385, -5.773502691896256e-09)),
    (0.3, 1e-12, (-0.7677936344934283, -0.7585090877696007, -0.9539392014169457, -3.144854510165755e-13)),
])
def test_hertz_stress_close_to_surface(r, z, reference):
    # Inside the contact close to the surface, the auxiliary variable u
    # was computed with cancellation
    np.testing.assert_allclose(np.ravel(Hertz.stress(np.array([r]), np.array([z]), 0.3)), reference, rtol=1e-12)


def test_greenwood_tripp_tolerance():
    # The tolerance applies to the solution, not to the change of the damped
    # iterate (which is smaller by the mixing factor)
    μ = 24.0425586841
    tol = 1e-6
    w, p, ρ = GT.GreenwoodTripp(1.0, μ, tol=tol)
    w_converged, p_converged, _ = GT.GreenwoodTripp(1.0, μ, tol=1e-10, maxiter=10000)
    assert np.abs(p - p_converged).max() < tol
    assert np.abs(w - w_converged).max() < tol
