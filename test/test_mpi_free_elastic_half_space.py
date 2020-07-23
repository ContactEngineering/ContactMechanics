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

from ContactMechanics import FreeFFTElasticHalfSpace
from NuMPI.Tools import Reduction


@pytest.fixture
def pnp(comm):
    return Reduction(comm)


@pytest.fixture
def basenpoints(comm):
    return (comm.Get_size() - 1) * 8
    # Base number of points in order to avoid
    # empty subdomains when using a lot of processors


@pytest.mark.parametrize("nx,ny", [(64, 32), (65, 33)])
def test_nb_grid_ptss(comm, pnp, nx, ny, basenpoints):
    nx += basenpoints
    ny += basenpoints
    sx, sy = 100, 200
    E_s = 3

    substrate = FreeFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                        fft='mpi', communicator=comm)
    assert substrate.nb_grid_pts == (nx, ny)
    assert substrate.nb_domain_grid_pts == (2 * nx, 2 * ny)
    assert pnp.sum(
        np.array(np.prod(substrate.nb_subdomain_grid_pts))) == 4 * nx * ny


@pytest.mark.parametrize("nx,ny", [(64, 32), (65, 33)])
def test_weights(comm, pnp, nx, ny, basenpoints):
    """
    Compare with the old serial Implementation
    """
    nx += basenpoints
    ny += basenpoints
    E_s = 1.5

    def _compute_fourier_coeffs_serial_impl(hs):
        """Compute the weights w relating fft(displacement) to fft(pressure):
           fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

           This version is less is copied from matscipy, use if memory is a
           concern
        """
        # pylint: disable=invalid-name
        facts = np.zeros(tuple((res * 2 for res in hs.nb_grid_pts)))
        a = hs._steps[0] * .5
        if hs.dim == 1:
            pass
        else:
            b = hs._steps[1] * .5
            x_s = np.arange(hs.nb_grid_pts[0] * 2)
            x_s = np.where(x_s <= hs.nb_grid_pts[0], x_s,
                           x_s - hs.nb_grid_pts[0] * 2) * hs._steps[0]
            x_s.shape = (-1, 1)
            y_s = np.arange(hs.nb_grid_pts[1] * 2)
            y_s = np.where(y_s <= hs.nb_grid_pts[1], y_s,
                           y_s - hs.nb_grid_pts[1] * 2) * hs._steps[1]
            y_s.shape = (1, -1)
            facts = 1 / (np.pi * hs.young) * ((x_s + a) * np.log(
                ((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +
                                     (x_s + a) * (x_s + a))) /
                ((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +
                                     (x_s + a) * (x_s + a)))) +
                                              (y_s + b) * np.log(
                ((x_s + a) + np.sqrt((y_s + b) * (y_s + b) +
                                     (x_s + a) * (x_s + a))) /
                ((x_s - a) + np.sqrt((y_s + b) * (y_s + b) +
                                     (x_s - a) * (x_s - a)))) +
                                              (x_s - a) * np.log(
                ((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +
                                     (x_s - a) * (x_s - a))) /
                ((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +
                                     (x_s - a) * (x_s - a)))) +
                                              (y_s - b) * np.log(
                ((x_s - a) + np.sqrt((y_s - b) * (y_s - b) +
                                     (x_s - a) * (x_s - a))) /
                ((x_s + a) + np.sqrt((y_s - b) * (y_s - b) +
                                     (x_s + a) * (x_s + a)))))
        weights = np.fft.rfftn(facts.T).T
        return weights

    sx, sy = 100, 200

    ref_weights = _compute_fourier_coeffs_serial_impl(
        FreeFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                fft="serial"))

    substrate = FreeFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                        fft='mpi', communicator=comm)
    local_weights = substrate._compute_greens_function()
    # print(local_weights.shape, ref_weights.shape, substrate.fourier_slices)
    np.testing.assert_allclose(local_weights,
                               ref_weights[substrate.fourier_slices], 1e-12)


@pytest.mark.parametrize("nx,ny", [(64, 32), (65, 33)])
def test_evaluate_disp_uniform_pressure(comm, pnp, nx, ny, basenpoints):
    nx += basenpoints
    ny += basenpoints

    sx, sy = 100, 200
    E_s = 1.5
    forces = np.zeros((2 * nx, 2 * ny))
    x = (np.arange(2 * nx) - (nx - 1) / 2) * sx / nx
    x.shape = (-1, 1)
    y = (np.arange(2 * ny) - (ny - 1) / 2) * sy / ny
    y.shape = (1, -1)

    forces[1:nx - 1, 1:ny - 1] = -np.ones((nx - 2, ny - 2)) * (sx * sy) / (
            nx * ny)
    a = (nx - 2) / 2 * sx / nx
    b = (ny - 2) / 2 * sy / ny
    refdisp = 1 / (np.pi * E_s) * (
            (x + a) * np.log(((y + b) + np.sqrt((y + b) * (y + b) +
                                                (x + a) * (x + a))) /
                             ((y - b) + np.sqrt((y - b) * (y - b) +
                                                (x + a) * (x + a)))) +
            (y + b) * np.log(((x + a) + np.sqrt((y + b) * (y + b) +
                                                (x + a) * (x + a))) /
                             ((x - a) + np.sqrt((y + b) * (y + b) +
                                                (x - a) * (x - a)))) +
            (x - a) * np.log(((y - b) + np.sqrt((y - b) * (y - b) +
                                                (x - a) * (x - a))) /
                             ((y + b) + np.sqrt((y + b) * (y + b) +
                                                (x - a) * (x - a)))) +
            (y - b) * np.log(((x - a) + np.sqrt((y - b) * (y - b) +
                                                (x - a) * (x - a))) /
                             ((x + a) + np.sqrt((y - b) * (y - b) +
                                                (x + a) * (x + a)))))

    substrate = FreeFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                        fft='mpi', communicator=comm)

    if comm.Get_size() > 1:
        with pytest.raises(FreeFFTElasticHalfSpace.Error):
            substrate.evaluate_disp(forces)
        with pytest.raises(FreeFFTElasticHalfSpace.Error):
            substrate.evaluate_disp(forces[nx, ny])

    # print(forces[substrate.subdomain_slices])
    computed_disp = substrate.evaluate_disp(forces[substrate.subdomain_slices])
    # print(computed_disp)
    # make the comparison only on the nonpadded domain
    s_c = tuple([slice(1, max(0, min(
        substrate.nb_grid_pts[i] - 1 - substrate.subdomain_locations[i],
        substrate.nb_subdomain_grid_pts[i])))
                 for i in range(substrate.dim)])

    s_refdisp = tuple([slice(s_c[i].start + substrate.subdomain_locations[i],
                             s_c[i].stop + substrate.subdomain_locations[i])
                       for i in range(substrate.dim)])
    # print(s_c)
    # print(s_refdisp)
    np.testing.assert_allclose(computed_disp[s_c], refdisp[s_refdisp])


def test_local_topography_subdomain_slices(comm):
    nx, ny = 64, 64

    np.random.seed(0)
    globaldata = np.random.random((nx, ny))

    substrate = FreeFFTElasticHalfSpace((nx, ny), 1., communicator=comm,
                                        fft='mpi')
    assert (globaldata[substrate.subdomain_slices]
            [substrate.local_topography_subdomain_slices]
            ==
            globaldata[substrate.topography_subdomain_slices]).all()
