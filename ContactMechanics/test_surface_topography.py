"""
Asserts computations of the elastic energy in SurfaceTopography are consistent with ContactMechanics
FFTElasticHalfSpace
"""
import numpy as np
import pytest
from SurfaceTopography.Container.SurfaceContainer import InMemorySurfaceContainer
from SurfaceTopography.Models.SelfAffine import SelfAffine
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="tests only serial functionalities, please execute with pytest")


@pytest.mark.parametrize(
    "shortcut_wavelength, hurst_exponent",
    [
        (2e-6, 0.8),
        (2e-6, 0.5),
        (2e-6, 0.3),
        (1e-7, 0.1),
        (1e-7, 0.5),
        (1e-7, 0.9),
        (1e-7, 1.),
    ]
)
def test_variance_half_derivative(shortcut_wavelength, hurst_exponent):
    from ContactMechanics import PeriodicFFTElasticHalfSpace
    n_pixels = 1024
    physical_size = .5e-4
    pixel_size = physical_size / n_pixels

    # test rolloff
    model_psd = SelfAffine(**{
        'cr': 5e-27,
        'shortcut_wavelength': shortcut_wavelength,
        'rolloff_wavelength': 2e-6,
        'hurst_exponent': hurst_exponent})

    Es = 1e6 / (1 - 0.5 ** 2)
    roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
    })

    # deterministic, brute force computation of the elastic energy
    hs = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=roughness.nb_grid_pts,
        young=Es,
        physical_sizes=roughness.physical_sizes)

    forces = hs.evaluate_force(roughness.heights())

    # Elastic energy per surface area
    Eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)

    Eel_analytic = Es / 4 * model_psd.variance_derivative(0.5)
    print(Eel_brute_force)
    print(Eel_analytic)

    np.testing.assert_allclose(Eel_analytic, Eel_brute_force, rtol=1e-1)


@pytest.mark.parametrize(
    "shortcut_wavelength, hurst_exponent",
    [
        (2e-6, 0.8),
        (2e-6, 0.5),
        (2e-6, 0.3),
        (1e-7, 0.1),
        (1e-7, 0.5),
        (1e-7, 0.9),
        (1e-7, 1.),
    ]
)
def test_variance_half_derivative_topography_and_container(shortcut_wavelength, hurst_exponent):
    """
    Tests different methods to compute the variance of the half-deriavative against
    the elastic energy in a contact mechanics simulation in full contact
    """
    from ContactMechanics import PeriodicFFTElasticHalfSpace
    unit = "m"

    n_pixels = 2048  # We need a good discretisation
    physical_size = .5e-4
    pixel_size = physical_size / n_pixels

    # test rolloff
    model_psd = SelfAffine(**{
        'cr': 5e-27,
        'shortcut_wavelength': shortcut_wavelength,
        'rolloff_wavelength': 2e-6,
        'hurst_exponent': hurst_exponent,
        'unit': unit})

    Es = 1e6 / (1 - 0.5 ** 2)
    roughness = model_psd.generate_roughness(**{
        'seed': 1,
        'n_pixels': n_pixels,
        'pixel_size': pixel_size,
    })

    # deterministic, brute force computation of the elastic energy
    hs = PeriodicFFTElasticHalfSpace(
        nb_grid_pts=roughness.nb_grid_pts,
        young=Es,
        physical_sizes=roughness.physical_sizes)

    forces = hs.evaluate_force(roughness.heights())

    # Elastic energy per surface area
    eel_brute_force = hs.evaluate_elastic_energy(forces, roughness.heights()) / np.prod(roughness.physical_sizes)

    eel_analytic = Es / 4 * model_psd.variance_derivative(0.5)

    eel_2d_psd = Es / 4 * roughness.moment_power_spectrum(order=1)

    # eel_1d_psd =
    eel_from_acf_profile = Es / 4 * roughness.variance_half_derivative_via_autocorrelation_from_profile()
    eel_from_acf_area = Es / 4 * roughness.variance_half_derivative_via_autocorrelation_from_area()

    c = InMemorySurfaceContainer([roughness, ])
    eel_container_psd = Es / 4 * c.ciso_moment(unit=unit)
    eel_container_acf = Es / 4 * c.variance_half_derivative_from_autocorrelation(unit=unit)

    print("brute force contact mechanics:", eel_brute_force)
    print("analytic from PSD:", eel_analytic)
    print("realisation 2d PSD", eel_2d_psd)

    print("realisation profile ACF:", eel_from_acf_profile)
    print("realisation areal ACF:", eel_from_acf_area)

    print("container PSD", eel_container_psd)
    print("container acf", eel_container_acf)

    np.testing.assert_allclose(eel_analytic, eel_brute_force, rtol=1e-1)
    np.testing.assert_allclose(eel_2d_psd, eel_brute_force, rtol=1e-1)
    np.testing.assert_allclose(eel_from_acf_profile, eel_brute_force, rtol=1e-1)
    np.testing.assert_allclose(eel_from_acf_area, eel_brute_force, rtol=1e-1)

    np.testing.assert_allclose(eel_container_psd, eel_brute_force, rtol=1e-1)
    np.testing.assert_allclose(eel_container_acf, eel_brute_force, rtol=1e-1)
