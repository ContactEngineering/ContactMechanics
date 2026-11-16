"""
Tests of the interface and error handling of the FFT-based half-spaces.
"""

import numpy as np
import pytest
from muGrid import FFTEngine

from ContactMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from ContactMechanics.FFTElasticHalfSpace import SemiPeriodicFFTElasticHalfSpace


def test_invalid_arguments():
    with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
        PeriodicFFTElasticHalfSpace((4, 4, 4), 1., (1., 1., 1.))
    with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
        PeriodicFFTElasticHalfSpace((4, 4), 1., (1., 1.), stiffness_q0=1., thickness=1.)
    with pytest.raises(ZeroDivisionError):
        PeriodicFFTElasticHalfSpace((0, 4), 1., (1., 1.))


def test_external_fft_engine():
    engine = FFTEngine([8, 6])
    substrate = PeriodicFFTElasticHalfSpace((8, 6), 1., (1., 1.), fftengine=engine)
    assert substrate.fftengine is engine
    with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
        PeriodicFFTElasticHalfSpace((8, 8), 1., (1., 1.), fftengine=engine)


def test_repr():
    assert 'E* = 2.0' in repr(PeriodicFFTElasticHalfSpace((8, 6), 2., (1., 1.)))


def test_stiffness_q0_1d():
    substrate = PeriodicFFTElasticHalfSpace((16,), 1., (1.,), stiffness_q0=3.)
    # A uniform pressure p leads to a uniform displacement p / stiffness_q0;
    # a unit force per pixel corresponds to p = 1 / area_per_pt
    displacements = substrate.evaluate_disp(-np.ones(16))
    np.testing.assert_allclose(displacements, 1 / substrate.area_per_pt / 3)


def test_shape_errors():
    substrate = PeriodicFFTElasticHalfSpace((8, 6), 1., (1., 1.))
    for method in [substrate.evaluate_disp, substrate.evaluate_force, substrate.evaluate_k_disp,
                   substrate.evaluate_k_force]:
        with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
            method(np.zeros((5, 5)))
    free = FreeFFTElasticHalfSpace((8, 6), 1., (1., 1.))
    with pytest.raises(FreeFFTElasticHalfSpace.Error):
        free.evaluate_disp(np.zeros((5, 5)))


def test_evaluate_in_fourier_space():
    substrate = PeriodicFFTElasticHalfSpace((8, 6), 1., (1., 1.))
    disp = np.random.default_rng(0).random((8, 6))
    energy, force = substrate.evaluate(disp, pot=True, forces=True)
    energy_without_forces, _ = substrate.evaluate(disp, pot=True, forces=False)
    np.testing.assert_allclose(energy_without_forces, energy)
    # Fourier transform in the layout of the FFT engine
    substrate.real_buffer.p[...] = disp
    substrate.fftengine.fft(substrate.real_buffer, substrate.fourier_buffer)
    disp_k = substrate.fourier_buffer.p.copy()
    energy_k, force_k = substrate.evaluate_k(disp_k, pot=True, forces=True)
    np.testing.assert_allclose(energy_k, energy)
    np.testing.assert_allclose(substrate.evaluate_k(disp_k, pot=True, forces=False)[0], energy)
    np.testing.assert_allclose(force_k, substrate.evaluate_k_force(disp))
    # compute_k stores the result
    substrate.compute_k(disp_k, pot=True, forces=True)
    np.testing.assert_allclose(substrate.energy, energy)
    np.testing.assert_allclose(substrate.force_k, force_k)
    # Displacements in Fourier space
    np.testing.assert_allclose(substrate.evaluate_k_disp(force), disp_k, atol=1e-12)


def test_spawn_child():
    free = FreeFFTElasticHalfSpace((16, 8), 2., (2., 1.), poisson=0.3)
    child = free.spawn_child((8, 4))
    assert child.nb_grid_pts == (8, 4)
    np.testing.assert_allclose(child.physical_sizes, (1., 0.5))
    np.testing.assert_allclose(child.contact_modulus, free.contact_modulus)
    with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
        PeriodicFFTElasticHalfSpace((8, 6), 1., (1., 1.)).spawn_child((4, 4))


def test_check_boundaries_with_mask():
    free = FreeFFTElasticHalfSpace((8, 8), 1., (1., 1.), check_boundaries=True)
    force = np.ma.masked_array(np.zeros((8, 8)), mask=np.zeros((8, 8), dtype=bool))
    force[0, :] = 1.
    with pytest.raises(FreeFFTElasticHalfSpace.FreeBoundaryError):
        free.check(force)
    # Masked values on the boundary are fine
    force.mask[0, :] = True
    free.check(force)


def test_semi_periodic():
    substrate = SemiPeriodicFFTElasticHalfSpace((8, 6), 1., (1., 1.), periodicity=(False, True), n_images=2)
    assert substrate.nb_domain_grid_pts == (15, 6)
    # Displacements on the topography grid are padded automatically
    disp = np.random.default_rng(0).random((8, 6))
    padded = np.zeros(substrate.nb_domain_grid_pts)
    padded[:8, :] = disp
    np.testing.assert_allclose(substrate.evaluate_force(disp), substrate.evaluate_force(padded))
    # Real-space Green's function with the origin at the center
    green = substrate.get_G_real()
    assert green.shape == (15, 6)
    assert np.argmax(green[:, 3]) == 7
    x_slice, y_slice = substrate.get_G_real_slices()
    np.testing.assert_allclose(x_slice, green[:, 3])
    np.testing.assert_allclose(y_slice, green[7, :])
