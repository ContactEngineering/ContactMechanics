#
# Copyright 2016, 2020 Lars Pastewka
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

from NuMPI.Tools.Reduction import Reduction

from ContactMechanics.ReferenceSolutions.Westergaard import _pressure
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import Topography
from ContactMechanics import make_system


@pytest.mark.skip
def test_constrained_conjugate_gradients(comm, fftengine_type):
    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 3.56
    pnp = Reduction(comm)

    for nx, ny in [(16384, 8 * comm.Get_size())]:
        # if ny is too small, one processor will end
        # with one empty fourier subdomain, what is not supported
        # 16384 points are indeed needed to reach the relative error of 1e-2

        for disp0, normal_force in [(-0.9, None),
                                    (-0.1, None)]:  # (0.1, None),
            subs = PeriodicFFTElasticHalfSpace(
                (nx, ny), E_s, (sx, sy),
                fft="serial" if comm.Get_size() == 1 else "mpi",
                communicator=comm)
            profile = np.resize(np.cos(2 * np.pi * np.arange(nx) / nx),
                                (ny, nx))
            surface = Topography(
                profile.T, physical_sizes=(sx, sy),
                # nb_grid_pts=substrate.nb_grid_pts,
                decomposition='domain',
                subdomain_locations=subs.topography_subdomain_locations,
                nb_subdomain_grid_pts=subs.topography_nb_subdomain_grid_pts,
                communicator=subs.communicator)
            system = make_system(subs, surface)

            result = system.minimize_proxy(offset=disp0,
                                           external_force=normal_force,
                                           pentol=1e-12)
            # offset = result.offset
            forces = result.jac
            # displ = result.x[:forces.shape[0], :forces.shape[1]]
            converged = result.success
            assert converged

            # print(forces)
            # print(displ)

            x = np.arange(nx) * sx / nx
            mean_pressure = pnp.sum(forces) / np.prod(subs.physical_sizes)
            pth = mean_pressure * \
                _pressure(x / sx, mean_pressure=sx * mean_pressure / E_s)

            # symetrize the Profile
            pth[1:] = pth[1:] + pth[:0:-1]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(np.arange(nx)*sx/nx, profile)
            # plt.plot(x, displ[:, 0], 'r-')
            # plt.plot(x, surface[:, 0]+offset, 'k-')
            # plt.figure()
            # plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
            # plt.plot(x, pth, 'r-')
            # plt.show()
            error_mask = np.abs(
                (forces[:, 0] / subs.area_per_pt - pth[
                    subs.subdomain_slices[0]]) >= 1e-12 + 1e-2 * np.abs(
                    pth[subs.subdomain_slices[0]]))

            # np.testing.assert_allclose(forces[:, 0]/substrate.area_per_
            # pth[substrate.subdomain_slices[0]], rtol=1e-2, atol = 1e-12)
            assert np.count_nonzero(
                error_mask) == 0, "max relative diff at index {} with "\
                                  "ref = {}, computed= {}".format(
                np.arange(subs.nb_subdomain_grid_pts[0])[error_mask],
                pth[subs.subdomain_slices[0]][error_mask],
                forces[:, 0][error_mask] / subs.area_per_pt)
