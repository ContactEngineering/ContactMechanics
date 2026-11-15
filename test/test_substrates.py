"""
Tests of the substrate base class, the real-space Green's function of
nonperiodic substrates and edge cases of the FFT-based half-spaces.
"""

import numpy as np
import pytest

from ContactMechanics import ElasticSubstrate, FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from ContactMechanics.FFTElasticHalfSpace import SemiPeriodicFFTElasticHalfSpace, _love_kernel

# Reference values of the Love kernel, computed with 60-digit arithmetic
# (mpmath) from the closed-form expression: x, y, a, b, kernel
LOVE_KERNEL_REFERENCE = [
    (0, 0, 0.5, 0.5, 3.5254943480781721009),
    (0.3, -0.2, 0.5, 0.5, 3.1399622385310785466),
    (-3, 7, 0.5, 0.5, 0.13140099374763772628),
    (-100, 0, 0.5, 0.5, 0.010000041666302079148),
    (0, -1000, 0.5, 0.5, 0.0010000000416666630208),
    (-16383, 0, 0.5, 0.5, 0.000061038881777161654068),
    (-16383, -16383, 0.5, 0.5, 0.000043161007217324844846),
    (500000, -300000, 0.5, 0.5, 1.7149858514252985436e-6),
    (-150, 120, 0.5, 0.05, 0.00052057969476361635853),
    (-524, -210, 2.0, 0.7, 0.009920069690374785545),
]


@pytest.mark.parametrize('x, y, a, b, reference', LOVE_KERNEL_REFERENCE)
def test_love_kernel_accuracy(x, y, a, b, reference):
    # The closed form loses accuracy far from the origin by cancellation
    # (previously up to a relative error of 1e-4 on a 16384 grid)
    np.testing.assert_allclose(_love_kernel(x, y, a, b), reference, rtol=1e-9)


def test_love_kernel_symmetry():
    x = np.linspace(-3000, 3000, 101).reshape(-1, 1)
    y = np.linspace(-2000, 2000, 77).reshape(1, -1)
    kernel = _love_kernel(x, y, 0.5, 0.3)
    np.testing.assert_array_equal(kernel, _love_kernel(-x, y, 0.5, 0.3))
    np.testing.assert_array_equal(kernel, _love_kernel(x, -y, 0.5, 0.3))


def test_free_greens_function_symmetric():
    # The real-space Green's function on the padded grid must be even
    substrate = FreeFFTElasticHalfSpace((64, 48), 1., (1., 1.))
    substrate.fftengine.ifft(substrate.fourier_buffer, substrate.real_buffer)
    green = np.array(substrate.real_buffer.p) * substrate.fftengine.normalisation
    np.testing.assert_allclose(green[1:, :], green[:0:-1, :], rtol=1e-12)
    np.testing.assert_allclose(green[:, 1:], green[:, :0:-1], rtol=1e-12)


def test_geometry_in_base_class():
    substrate = PeriodicFFTElasticHalfSpace((16, 8), 1., physical_sizes=(2.,))
    assert isinstance(substrate, ElasticSubstrate)
    assert substrate.dim == 2
    assert substrate.nb_grid_pts == (16, 8)
    # The last physical size is repeated
    assert substrate.physical_sizes == (2., 2.)
    assert substrate.area_per_pt == 4. / 128


@pytest.mark.parametrize('substrate_class', [FreeFFTElasticHalfSpace, SemiPeriodicFFTElasticHalfSpace])
def test_poisson(substrate_class):
    forces = np.zeros((32, 32))
    forces[10:14, 12:16] = -1.
    poisson = 0.3
    displacements = substrate_class((32, 32), 1., (1., 1.), poisson=poisson).evaluate_disp(forces)
    reference = substrate_class((32, 32), 1. / (1 - poisson ** 2), (1., 1.)).evaluate_disp(forces)
    np.testing.assert_allclose(displacements, reference, rtol=1e-12)


def test_stiffness_q0_zero_1d():
    substrate = PeriodicFFTElasticHalfSpace((16,), 1., (1.,), stiffness_q0=0.)
    # A uniform force does not lead to a displacement
    np.testing.assert_allclose(substrate.evaluate_disp(np.ones(16)), 0, atol=1e-12)


def test_finite_thickness_requires_2d():
    with pytest.raises(PeriodicFFTElasticHalfSpace.Error):
        PeriodicFFTElasticHalfSpace((16,), 1., (1.,), thickness=1.)


@pytest.mark.parametrize('substrate_class', [FreeFFTElasticHalfSpace, SemiPeriodicFFTElasticHalfSpace])
def test_nonperiodic_requires_2d(substrate_class):
    with pytest.raises(substrate_class.Error):
        substrate_class((16,), 1., (1.,))


def test_incompressible_layer():
    substrate = PeriodicFFTElasticHalfSpace((8, 8), 1., (1., 1.), thickness=1., poisson=0.5)
    # Uniform pressure cannot compress an incompressible layer on a rigid
    # substrate
    np.testing.assert_allclose(substrate.evaluate_disp(np.ones((8, 8))), 0, atol=1e-12)
    # The response at finite wavevectors approaches the one of nearly
    # incompressible layers
    forces = np.zeros((8, 8))
    forces[2, 3] = 1.
    forces -= forces.mean()
    nearly = PeriodicFFTElasticHalfSpace((8, 8), 1., (1., 1.), thickness=1., poisson=0.4999999)
    np.testing.assert_allclose(substrate.evaluate_disp(forces), nearly.evaluate_disp(forces), rtol=1e-5, atol=1e-10)


def test_semi_periodic_image_sum_converges():
    # The sum over periodic images used to diverge logarithmically, i.e. the
    # mean displacement depended on the number of images
    forces = np.zeros((32, 32))
    forces[14:18, 14:18] = -1.
    reference = SemiPeriodicFFTElasticHalfSpace((32, 32), 1., (1., 1.), periodicity=(False, True),
                                                n_images=160).evaluate_disp(forces)
    errors = []
    for n_images in [10, 20, 40]:
        displacements = SemiPeriodicFFTElasticHalfSpace((32, 32), 1., (1., 1.), periodicity=(False, True),
                                                        n_images=n_images).evaluate_disp(forces)
        errors += [np.abs(displacements - reference).max() / np.abs(reference).max()]
    assert errors[-1] < 1e-5
    assert np.all(np.diff(errors) < 0)


def test_semi_periodic_real_space_greens_function(comm):
    # The real-space Green's function is available on the full grid, also
    # for domain-decomposed calculations
    from NuMPI import MPI
    serial = SemiPeriodicFFTElasticHalfSpace((16, 12), 1., (1., 1.), periodicity=(False, True), n_images=2,
                                             communicator=MPI.COMM_SELF)
    reference = serial.get_G_real()
    assert reference.shape == (31, 12)
    # G(0, 0) is at the center and is the maximum
    assert np.argmax(reference[:, 6]) == 15
    # It agrees with the stored (local) Green's function, reordered
    order_x = np.argsort(np.round(np.fft.fftfreq(31) * 31))
    order_y = np.argsort(np.round(np.fft.fftfreq(12) * 12))
    np.testing.assert_allclose(reference, serial.G_real[np.ix_(order_x, order_y)])
    parallel = SemiPeriodicFFTElasticHalfSpace((16, 12), 1., (1., 1.), periodicity=(False, True), n_images=2,
                                               communicator=comm, fft='mpi')
    np.testing.assert_allclose(parallel.get_G_real(), reference)
    x_slice, y_slice = parallel.get_G_real_slices()
    np.testing.assert_allclose(x_slice, reference[:, 6])
    np.testing.assert_allclose(y_slice, reference[15, :])


def test_external_fftengine_communicator(comm):
    # A substrate on an external (possibly distributed) FFT engine reduces
    # over the engine's processes, whatever communicator is passed
    from NuMPI import MPI
    reference = SemiPeriodicFFTElasticHalfSpace((16, 12), 1., (1., 1.), periodicity=(False, True), n_images=2,
                                                communicator=comm, fft='mpi')
    substrate = SemiPeriodicFFTElasticHalfSpace((16, 12), 1., (1., 1.), periodicity=(False, True), n_images=2,
                                                communicator=MPI.COMM_SELF, fftengine=reference.fftengine)
    assert substrate.communicator.size == comm.size
    assert substrate._zero_mode_shift == reference._zero_mode_shift
    np.testing.assert_allclose(substrate.greens_function, reference.greens_function)
    np.testing.assert_allclose(substrate.get_G_real(), reference.get_G_real())


@pytest.mark.parametrize('periodicity', [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize('physical_sizes', [(10., 1.), (1., 10.), (1., 1.)])
def test_semi_periodic_positive_energy(periodicity, physical_sizes):
    # The image sum fixes the response to a mean pressure only up to a
    # constant; the convention for it must keep the elastic energy positive,
    # also on elongated domains
    substrate = SemiPeriodicFFTElasticHalfSpace((32, 8), 1., physical_sizes, periodicity=periodicity, n_images=10)
    assert np.all(substrate.greens_function.real > 0)
    forces = -np.ones((32, 8))
    assert -np.sum(forces * substrate.evaluate_disp(forces)) > 0
    # Random (also tensile) forces
    forces = np.random.default_rng(0).standard_normal((32, 8))
    assert -np.sum(forces * substrate.evaluate_disp(forces)) > 0
