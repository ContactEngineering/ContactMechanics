#
# Copyright 2019-2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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

import numpy as np
import pytest

from muFFT import FFT
from NuMPI import MPI

from ContactMechanics import PeriodicFFTElasticHalfSpace

from NuMPI.Tools import Reduction


@pytest.fixture
def pnp(comm):
    return Reduction(comm)


@pytest.fixture
def basenpoints(comm):
    return (comm.Get_size() - 1) * 8
    # Base number of points in order to avoid empty subdomains
    # when using a lot of processors


@pytest.mark.parametrize("nx, ny", [(64, 33),
                                    (65, 32),
                                    (64, 64)])
# TODO: merge the serial test of the weights into this
def test_weights(comm, pnp, nx, ny,
                 basenpoints):
    """compares the MPI-Implemtation of the halfspace with the serial one"""
    nx += basenpoints
    ny += basenpoints
    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 1.0

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fft='mpi', communicator=comm)
    reference = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fft="fftw",
                                            communicator=MPI.COMM_SELF)
    np.testing.assert_allclose(
        reference.greens_function[substrate.fourier_slices],
        substrate.greens_function, rtol=0, atol=1e-16,
        err_msg="weights are different")
    np.testing.assert_allclose(
        reference.surface_stiffness[substrate.fourier_slices],
        substrate.surface_stiffness, rtol=0, atol=1e-16,
        err_msg="iweights are different")


@pytest.mark.parametrize("nx, ny", [(8, 15),
                                    (8, 4),
                                    (9, 4),
                                    (113, 765)])
def test_sineWave_disp(comm, pnp, nx, ny, basenpoints):
    """
    for given sinusoidal displacements, compares the pressures and the energies
    to the analytical solutions

    Special cases at the edges of the fourier domain are done

    Parameters
    ----------
    comm
    pnp
    fftengine_class
    nx
    ny
    basenpoints

    Returns
    -------

    """
    nx += basenpoints
    ny += basenpoints
    sx = 2.45  # 30.0
    sy = 1.0
    ATOL = 1e-10 * (nx * ny)
    # equivalent Young's modulus
    E_s = 1.0

    for k in [(1, 0),
              (0, 1),
              (1, 2),
              (nx // 2, 0),
              (1, ny // 2),
              (0, 2),
              (nx // 2, ny // 2),
              (0, ny // 2)]:
        # print("testing wavevector ({}* np.pi * 2 / sx,
        # {}* np.pi * 2 / sy) ".format(*k))
        qx = k[0] * np.pi * 2 / sx
        qy = k[1] * np.pi * 2 / sy
        q = np.sqrt(qx ** 2 + qy ** 2)

        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
                           np.linspace(0, sx, nx + 1)[:-1])
        disp = np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)

        refpressure = - disp * E_s / 2 * q

        substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                fft='mpi', communicator=comm)
        fftengine = FFT((nx, ny), fft='mpi', communicator=comm)
        fftengine.create_plan(1)

        kpressure = substrate.evaluate_k_force(
            disp[substrate.subdomain_slices]) / substrate.area_per_pt
        expected_k_disp = np.zeros((nx // 2 + 1, ny), dtype=complex)
        expected_k_disp[k[0], k[1]] += (.5 - .5j)*(nx * ny)

        # add the symetrics
        if k[0] == 0:
            expected_k_disp[0, -k[1]] += (.5 + .5j)*(nx * ny)
        if k[0] == nx // 2 and nx % 2 == 0:
            expected_k_disp[k[0], -k[1]] += (.5 + .5j)*(nx * ny)

        fft_disp = np.zeros(substrate.nb_fourier_grid_pts, order='f',
                            dtype=complex)
        fftengine.fft(disp[substrate.subdomain_slices], fft_disp)
        np.testing.assert_allclose(fft_disp,
                                   expected_k_disp[substrate.fourier_slices],
                                   rtol=1e-7, atol=ATOL)

        expected_k_pressure = - E_s / 2 * q * expected_k_disp
        np.testing.assert_allclose(
            kpressure, expected_k_pressure[substrate.fourier_slices],
            rtol=1e-7, atol=ATOL)

        computedpressure = substrate.evaluate_force(
            disp[substrate.subdomain_slices]) / substrate.area_per_pt
        np.testing.assert_allclose(computedpressure,
                                   refpressure[substrate.subdomain_slices],
                                   atol=ATOL, rtol=1e-7)

        computedenergy_kspace = \
            substrate.evaluate(disp[substrate.subdomain_slices], pot=True,
                               forces=False)[0]
        computedenergy = \
            substrate.evaluate(disp[substrate.subdomain_slices], pot=True,
                               forces=True)[0]
        refenergy = E_s / 8 * 2 * q * sx * sy

        # print(substrate.nb_domain_grid_pts[-1] % 2)
        # print(substrate.nb_fourier_grid_pts)
        # print(substrate.fourier_locations[-1] +
        # substrate.nb_fourier_grid_pts[-1] - 1)
        # print(substrate.nb_domain_grid_pts[-1] // 2 )
        # print(computedenergy)
        # print(computedenergy_kspace)
        # print(refenergy)
        np.testing.assert_allclose(
            computedenergy, refenergy, rtol=1e-10,
            err_msg="wavevektor {} for nb_domain_grid_pts {}, "
                    "subdomain nb_grid_pts {}, nb_fourier_grid_pts {}"
                    .format(k, substrate.nb_domain_grid_pts,
                            substrate.nb_subdomain_grid_pts,
                            substrate.nb_fourier_grid_pts))
        np.testing.assert_allclose(
            computedenergy_kspace, refenergy,
            rtol=1e-10,
            err_msg="wavevektor {} for nb_domain_grid_pts {}, "
                    "subdomain nb_grid_pts {}, nb_fourier_grid_pts {}"
                    .format(k, substrate.nb_domain_grid_pts,
                            substrate.nb_subdomain_grid_pts,
                            substrate.nb_fourier_grid_pts))


@pytest.mark.parametrize("nx, ny", [(8, 8),
                                    (17, 128),
                                    (3, 128)])
def test_sineWave_disp_rotation_invariance(comm, pnp, nx, ny, basenpoints):
    """
    for a sinusoidal displacement, test if the energy depends on if the wave is
    oriented in x or y direction

    Parameters
    ----------
    comm
    pnp
    fftengine_class
    nx
    ny
    basenpoints

    Returns
    -------

    """
    nx += basenpoints
    ny += basenpoints
    sx = 3.  # 30.0
    sy = 3.

    # equivalent Young's modulus
    E_s = 1.0

    computedenergies = []
    computedenergies_kspace = []
    for k in [(min(nx, ny) // 2, 0), (0, min(nx, ny) // 2)]:
        qx = k[0] * np.pi * 2 / sx
        qy = k[1] * np.pi * 2 / sy

        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
                           np.linspace(0, sx, nx + 1)[:-1])
        # At the Nyquist frequency for even number of points, the energy
        # computation can only be exact for this point
        disp = np.cos(qx * X + qy * Y) + np.sin(
            qx * X + qy * Y)

        substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                fft='mpi', communicator=comm)

        computedenergies_kspace += [
            substrate.evaluate(disp[substrate.subdomain_slices], pot=True,
                               forces=False)[0]]
        computedenergies += [
            substrate.evaluate(disp[substrate.subdomain_slices], pot=True,
                               forces=True)[0]]

    # np.testing.assert_allclose(computedpressures[0],computedpressures[1].T)
    np.testing.assert_allclose(*computedenergies, rtol=1e-10)
    np.testing.assert_allclose(*computedenergies_kspace, rtol=1e-10)


@pytest.mark.parametrize("nx, ny", [(64, 33),
                                    (65, 32),
                                    (64, 64)])
def test_sineWave_force(comm, pnp, nx, ny, basenpoints):
    """
    for  a given sinusoidal force, compares displacement with a reference
    solution

    Parameters
    ----------
    comm
    pnp
    fftengine_class
    nx
    ny
    basenpoints

    Returns
    -------

    """
    nx += basenpoints
    ny += basenpoints
    sx = 2  # 30.0
    sy = 1.0

    # equivalent Young's modulus
    E_s = 1.0

    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
                       np.linspace(0, sx, nx + 1)[:-1])

    qx = 1 * np.pi * 2 / sx
    qy = 4 * np.pi * 2 / sy

    q = np.sqrt(qx ** 2 + qy ** 2)
    p = np.cos(qx * X + qy * Y)

    refdisp = - p / E_s * 2 / q

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fft='mpi', communicator=comm)
    computeddisp = substrate.evaluate_disp(
        p[substrate.subdomain_slices] * substrate.area_per_pt)
    np.testing.assert_allclose(computeddisp,
                               refdisp[substrate.subdomain_slices], atol=1e-7,
                               rtol=1e-10)

    # computedenergy = substrate.evaluate(p[substrate.subdomain_slices]*
    # substrate.area_per_pt)

    # refenergy = sx * sy/(2 * q * E_s) * 1
    # np.testing.assert_allclose(computedenergy,refenergy,rtol = 1e-4)


#    def test_k_force_maxq(self):
#        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
#        np.linspace(0, sx, nx + 1)[:-1])
#
#        qx = 1 * np.pi * 2 / sx
#        qy = ny//2 * np.pi * 2 / sy
#
#        q = np.sqrt(qx ** 2 + qy ** 2)
#        h=1
#        disp = h*np.cos(qx * X + qy * Y)
#
#        ref_k_force= np.zeros((nx, ny//2+1))
#        ref_k_force[1,ny//2] = q * h *E_s /2

@pytest.mark.parametrize("nx, ny", [(64, 33),
                                    (65, 32),
                                    (64, 64)])
def test_multipleSineWaves_evaluate(comm, pnp, nx, ny, basenpoints):
    """
    displacements: superposition of sinwaves, compares forces and energes with
    analytical solution

    Parameters
    ----------
    comm
    pnp
    fftengine_class
    nx
    ny
    basenpoints

    Returns
    -------

    """
    nx += basenpoints
    ny += basenpoints
    sx = 2  # 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 1.0

    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1],
                       np.linspace(0, sx, nx + 1)[:-1])

    disp = np.zeros((nx, ny))
    refForce = np.zeros((nx, ny))

    refEnergy = 0
    for qx, qy in zip((1, 0, 5, nx // 2 - 1),
                      (4, 4, 0, ny // 2 - 2)):
        qx = qx * np.pi * 2 / sx
        qy = qy * np.pi * 2 / sy

        q = np.sqrt(qx ** 2 + qy ** 2)
        h = 1  # q**(-0.8)
        disp += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y))
        refForce += h * (np.cos(qx * X + qy * Y) + np.sin(
            qx * X + qy * Y)) * E_s / 2 * q
        refEnergy += E_s / 8 * q * 2 * h ** 2
        # * 2 because the amplitude of cos(x) + sin(x) is sqrt(2)

    # max possible Wavelengths at the edge

    for qx, qy in zip((nx // 2, nx // 2, 0),
                      (ny // 2, 0, ny // 2)):
        qx = qx * np.pi * 2 / sx
        qy = qy * np.pi * 2 / sy

        q = np.sqrt(qx ** 2 + qy ** 2)
        h = 1  # q**(-0.8)
        disp += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y))
        refForce += h * (np.cos(qx * X + qy * Y) + np.sin(
            qx * X + qy * Y)) * E_s / 2 * q

        refEnergy += E_s / 8 * q * h ** 2 * 2
        # * 2 because the amplitude of cos(x) + sin(x) is sqrt(2)

    refEnergy *= sx * sy
    refForce *= -sx * sy / (nx * ny)

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fft='mpi', communicator=comm)
    computed_E_k_space = substrate.evaluate(disp[substrate.subdomain_slices],
                                            pot=True, forces=False)[0]
    # If force is not queried this computes the energy using kspace
    computed_E_realspace, computed_force = substrate.evaluate(
        disp[substrate.subdomain_slices], pot=True,
        forces=True)

    # print("{}: Local: E_kspace: {}, E_realspace: {}"
    # .format(substrate.fftengine.comm.Get_rank(),computed_E_k_space,computed_E_realspace))
    # print(computed_E_k_space)
    # print(refEnergy)

    # if substrate.fftengine.comm.Get_rank() == 0 :
    #    print(computed_E_k_space)
    #    print(computed_E_realspace)

    # print("{}: Global: E_kspace: {}, E_realspace: {}"
    # .format(substrate.fftengine.comm.Get_rank(),
    # computed_E_k_space, computed_E_realspace))

    # Make an MPI-Reduce of the Energies !
    # print(substrate.evaluate_elastic_energy(refForce, disp))
    # print(0.5*np.vdot(refForce,disp))
    # print(substrate.evaluate_elastic_energy(substrate.evaluate_force(disp),disp))
    # print(computed_E_k_space)
    # print(computed_E_realspace)
    # print(refEnergy)

    np.testing.assert_almost_equal(computed_E_k_space, refEnergy)
    np.testing.assert_almost_equal(computed_E_realspace, refEnergy)
    np.testing.assert_allclose(computed_force,
                               refForce[substrate.subdomain_slices], atol=1e-7,
                               rtol=1e-10)
