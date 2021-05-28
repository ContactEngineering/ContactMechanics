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

"""
Tests adhesion-free flat punch results
"""

import numpy as np
import pytest
import scipy.optimize as optim

from NuMPI import MPI

from ContactMechanics.ReferenceSolutions.Westergaard import _pressure
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import Topography, UniformLineScan
from ContactMechanics import make_system


def test_constrained_conjugate_gradients():
    # system physical_sizes
    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 3.56
    for nx, ny in [(256, 16)]:  # , (256, 15), (255, 16)]:
        for disp0, normal_force in [(-0.9, None),
                                    (-0.1, None)]:  # (0.1, None),
            substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s,
                                                    (sx, sy))
            profile = np.resize(np.cos(2 * np.pi * np.arange(nx) / nx),
                                (ny, nx))
            surface = Topography(profile.T, (sx, sy))
            system = make_system(substrate, surface)

            result = system.minimize_proxy(offset=disp0,
                                           external_force=normal_force,
                                           pentol=1e-9)
            # offset = result.offset
            forces = result.jac
            # displ = result.x[:forces.shape[0], :forces.shape[1]]
            converged = result.success
            assert converged

            x = np.arange(nx) * sx / nx
            mean_pressure = np.mean(forces) / substrate.area_per_pt
            pth = mean_pressure * _pressure(
                x / sx,
                mean_pressure=sx * mean_pressure / E_s)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(np.arange(nx)*self.sx/nx, profile)
            # plt.plot(x, displ[:, 0], 'r-')
            # plt.plot(x, surface[:, 0]+offset, 'k-')
            # plt.figure()
            # plt.plot(x, forces[:, 0]/substrate.area_per_pt, 'k-')
            # plt.plot(x, pth, 'r-')
            # plt.show()
            assert (np.allclose(
                forces[:nx // 2, 0] /
                substrate.area_per_pt, pth[:nx // 2], atol=1e-2))


@pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                    reason="test only serial functionalities, "
                           "please execute with pytest")
@pytest.mark.parametrize("dx,n", [(1., 32),
                                  (1., 33),
                                  (0.5, 32)])
def test_lbfgsb_1D(dx, n):
    # test the 1D periodic objective is working

    s = n * dx

    Es = 1.

    substrate = PeriodicFFTElasticHalfSpace((n,), young=Es,
                                            physical_sizes=(s,), fft='serial')

    surface = UniformLineScan(np.cos(2 * np.pi * np.arange(n) / n), (s,))
    system = make_system(substrate, surface)

    offset = 0.005
    lbounds = np.zeros(n)
    bnds = system._reshape_bounds(lbounds, )
    init_gap = np.zeros(n, )  # .flatten()
    disp = init_gap + surface.heights() + offset

    res = optim.minimize(system.primal_objective(offset, gradient=True),
                         disp,
                         method='L-BFGS-B', jac=True,
                         bounds=bnds,
                         options=dict(gtol=1e-5 * Es * surface.rms_slope_from_profile()
                                      * surface.area_per_pt,
                                      ftol=0))

    forces = res.jac
    gap = res.x
    converged = res.success
    assert converged
    if hasattr(res.message, "decode"):
        decoded_message = res.message.decode()
    else:
        decoded_message = res.message

    assert decoded_message == \
        'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'

    x = np.arange(n) * s / n
    mean_pressure = np.mean(forces) / substrate.area_per_pt

    assert mean_pressure > 0

    pth = mean_pressure * _pressure(
        x / s,
        mean_pressure=s * mean_pressure / Es)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        # plt.plot(np.arange(n)*s/n, surface.heights())
        plt.plot(x, gap + (surface.heights()[:] + offset), 'r-')
        plt.plot(x, (surface.heights()[:] + offset), 'k-')
        plt.figure()
        plt.plot(x, forces / substrate.area_per_pt, 'k-')
        plt.plot(x, pth, 'r-')
        plt.show()

    assert (np.allclose(
        forces /
        substrate.area_per_pt, pth, atol=1e-2))
