#
# Copyright 2016, 2018, 2020 Lars Pastewka
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

from ContactMechanics import FreeFFTElasticHalfSpace
from SurfaceTopography import Topography
from ContactMechanics import make_system

import pytest


@pytest.mark.parametrize("nx", [255, 256])
@pytest.mark.parametrize("ny", [255, 256])
@pytest.mark.parametrize("disp0, normal_force", [(None, 15), (0.1, None)])
def test_constrained_conjugate_gradients(nx, ny, disp0, normal_force):
    # punch radius:
    r_s = 20.0
    # equivalent Young's modulus
    E_s = 3.56

    sx = sy = 2.5 * r_s
    substrate = FreeFFTElasticHalfSpace((nx, ny), E_s, (sx, sy))
    x_range = np.arange(nx).reshape(-1, 1)
    y_range = np.arange(ny).reshape(1, -1)
    r_sq = (sx / nx * (x_range - nx // 2)) ** 2 + \
           (sy / ny * (y_range - ny // 2)) ** 2
    surface = Topography(
        np.ma.masked_where(r_sq > r_s ** 2, np.zeros([nx, ny])),
        (sx, sy))
    system = make_system(substrate, surface)
    try:
        result = system.minimize_proxy(offset=disp0,
                                       external_force=normal_force,
                                       pentol=1e-4)
    except substrate.FreeBoundaryError as err:
        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            # ax.pcolormesh(substrate.force /
            # surface.area_per_pt,rasterized=True)
            plt.colorbar(ax.pcolormesh(surface.heights(),
                                       rasterized=True))
            ax.set_xlabel("")
            ax.set_ylabel("")

            ax.legend()

            fig.tight_layout()
            plt.show(block=True)

        raise err
    offset = result.offset
    forces = -result.jac
    converged = result.success

    # Check that calculation has converged
    assert converged

    # Check that target values have been reached
    if disp0 is not None:
        np.testing.assert_almost_equal(offset, disp0)
    if normal_force is not None:
        np.testing.assert_almost_equal(-forces.sum(), normal_force)

    # Check contact stiffness
    np.testing.assert_almost_equal(
        -forces.sum() / offset / (2 * r_s * E_s),
        1.0, decimal=2)
