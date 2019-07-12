#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

if _withMPI:
    from NuMPI.Tools.Reduction import Reduction

import numpy as np
from PyCo.ContactMechanics import HardWall
from PyCo.ReferenceSolutions.Westergaard import _pressure
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
from PyCo.Topography import Topography
from PyCo.System import make_system
from PyCo.Tools.Logger import screen


from PyCo.Tools.Logger import Logger


# -----------------------------------------------------------------------------
def test_constrained_conjugate_gradients(comm, fftengine_type):
    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 3.56
    pnp = Reduction(comm)

    for nx, ny in [(16384, 8 * comm.Get_size())]:  # , (256, 15), (255, 16)]: #256,16
        # if ny is too small, one processor will end
        # with one empty fourier subdomain, what is not supported
        # 16384 points are indeed needed to reach the relative error of 1e-2

        for disp0, normal_force in [(-0.9, None), (-0.1, None)]:  # (0.1, None),
            substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                    fft="mpi", communicator=comm)
            interaction = HardWall()
            profile = np.resize(np.cos(2 * np.pi * np.arange(nx) / nx), (ny, nx))
            surface = Topography(profile.T, physical_sizes=(sx, sy),
                                 # nb_grid_pts=substrate.nb_grid_pts,
                                 subdomain_locations=substrate.topography_subdomain_locations,
                                 nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
                                 communicator=substrate.communicator)
            system = make_system(substrate, interaction, surface)

            result = system.minimize_proxy(offset=disp0,
                                           external_force=normal_force,
                                           pentol=1e-12)
            offset = result.offset
            forces = result.jac
            displ = result.x[:forces.shape[0], :forces.shape[1]]
            converged = result.success
            assert converged

            # print(forces)
            # print(displ)

            x = np.arange(nx) * sx / nx
            mean_pressure = pnp.sum(forces) / np.prod(substrate.physical_sizes)
            pth = mean_pressure * _pressure(x / sx, mean_pressure=sx * mean_pressure / E_s)

            # symetrize the Profile
            pth[1:] = pth[1:] + pth[:0:-1]

            # import matplotlib.pyplot as plt
            # plt.figure()
            ##plt.plot(np.arange(nx)*sx/nx, profile)
            # plt.plot(x, displ[:, 0], 'r-')
            # plt.plot(x, surface[:, 0]+offset, 'k-')
            # plt.figure()
            # plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
            # plt.plot(x, pth, 'r-')
            # plt.show()
            error_mask = np.abs(
                (forces[:, 0] / substrate.area_per_pt - pth[substrate.subdomain_slices[0]]) >= 1e-12 + 1e-2 * np.abs(
                    pth[substrate.subdomain_slices[0]]))

            # np.testing.assert_allclose(forces[:, 0]/substrate.area_per_ pth[substrate.subdomain_slices[0]], rtol=1e-2, atol = 1e-12)
            assert np.count_nonzero(
                error_mask) == 0, "max relative diff at index {} with ref = {}, computed= {}".format(
                np.arange(substrate.nb_subdomain_grid_pts[0])[error_mask], pth[substrate.subdomain_slices[0]][error_mask],
                forces[:, 0][error_mask] / substrate.area_per_pt)
