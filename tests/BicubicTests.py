#
# Copyright 2020 Antoine Sanner
#           2020 Lars Pastewka
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

"""
Tests for the bicubic interpolation module
"""

import pytest

import numpy as np

from PyCo.ContactMechanics.Tools.Interpolation import Bicubic
from PyCo.SurfaceTopography.Generation import fourier_synthesis

nx = 17
ny = 22

def test_grid_values(tol=1e-9):
    field = np.random.random([nx, ny])
    interp = Bicubic(field)
    for i in range(nx):
        for j in range(ny):
            assert abs(interp(i, j) - field[i, j]) < tol

    x, y = np.mgrid[:nx, :ny]

    for der in [0, 1, 2]:
        if der == 0:
            interp_field = interp(x, y, derivative=der)
        elif der == 1:
            interp_field, _, _ = interp(x, y, derivative=der)
        else:
            interp_field, _, _, _, _, _ = interp(x, y, derivative=der)
        assert np.allclose(interp_field, field)


def test_wrong_derivative(tol=1e-9):
    field = np.random.random([nx, ny])
    interp = Bicubic(field)
    with pytest.raises(ValueError):
        interp(1, 1, derivative=3)
    with pytest.raises(ValueError):
        interp(1, 1, derivative=-1)


def test_wrong_grid(tol=1e-9):
    field = np.random.random([nx, ny])
    derx = np.random.random([nx, ny-1])
    dery = np.random.random([nx, ny])
    with pytest.raises(ValueError):
        Bicubic(field, derx, dery)


def test_grid_derivatives(tol=1e-9):
    field = np.random.random([nx, ny])
    derx = np.random.random([nx, ny])
    dery = np.random.random([nx, ny])
    interp = Bicubic(field, derx, dery)
    for i in range(nx):
        for j in range(ny):
            assert abs(interp(i, j) - field[i, j]) < tol

    x, y = np.mgrid[:nx, :ny]

    interp_field, interp_derx, interp_dery = interp(x, y, derivative=1)
    assert np.allclose(interp_field, field)
    assert np.allclose(interp_derx, derx)
    assert np.allclose(interp_dery, dery)

def test_bicubic_between(plot=False):
    # grid points based on which the interpolation is made
    nx = 32
    ny = 17
    sx = nx
    sy = ny

    x = np.arange(nx).reshape(-1, 1)
    y = np.arange(ny).reshape(1, -1)

    fun = lambda x, y: np.sin(2 * np.pi  *  x / sx) * np.cos(2 * np.pi  *  y / sy)
    dfun_dx = lambda x, y: 2 * np.pi / sx * np.cos(2 * np.pi  *  x / sx) * np.cos(2 * np.pi  *  y / sy)
    dfun_dy = lambda x, y: -2 * np.pi / sy * np.sin(2 * np.pi  *  x / sx) * np.sin(2 * np.pi  *  y / sy)

    interp = Bicubic(fun(x, y), dfun_dx(x, y), dfun_dy(x, y))
    interp_field, interp_derx, interp_dery = interp(x * np.ones_like(y), y * np.ones_like(x), derivative=1)

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax, axdx, axdy) = plt.subplots(3,1)
        ax.plot(x,  fun(x, y)[:, 0], "k+")
        ax.plot(x, interp_field[:,0], "ko", mfc="none")
        axdx.plot(x, dfun_dx(x, y)[:, 0], "r+")
        axdx.plot(x, interp_derx[:,0], "ro", mfc="none")
        axdy.plot(y.flat, dfun_dy(x, y)[0,:], "g+")
        axdy.plot(y.flat, interp_dery[0,:], "go", mfc="none")

    # interpolate between the points used for definition
    fac = 8
    x_fine = np.arange(fac * nx).reshape(-1, 1) / fac
    y_fine = np.arange(fac * ny).reshape(1, -1) / fac

    interp_field0 = interp(x_fine * np.ones_like(y_fine), y_fine * np.ones_like(x_fine), derivative=0)
    interp_field1, interp_derx1, interp_dery1 = interp(x_fine * np.ones_like(y_fine), y_fine * np.ones_like(x_fine),
                                                       derivative=1)
    interp_field2, interp_derx2, interp_dery2, interp_derxx2, interp_deryy2, interp_derxy2 = \
        interp(x_fine * np.ones_like(y_fine), y_fine * np.ones_like(x_fine), derivative=2)
    if plot:
        ax.plot(x_fine, interp_field1[:,0], 'k-')
        axdx.plot(x_fine.flat, interp_derx1[:,0], 'r-')
        axdy.plot(y_fine.flat, interp_dery1[0,:], 'g-')
        fig.show()

    np.testing.assert_allclose(interp_field0, fun(x_fine, y_fine), atol = 1e-2)
    np.testing.assert_allclose(interp_field1, fun(x_fine, y_fine), atol = 1e-2)
    np.testing.assert_allclose(interp_derx1, dfun_dx(x_fine, y_fine), atol = 1e-2)
    np.testing.assert_allclose(interp_dery1, dfun_dy(x_fine, y_fine), atol = 1e-2)
    np.testing.assert_allclose(interp_field2, fun(x_fine, y_fine), atol = 1e-2)

@pytest.mark.parametrize("sx, sy", [(5.,6.), (50.,60.)])
def test_wrapped_bicubic_vs_fourier(sx, sy):
    # test against fourier interpolation
    # sx and sy are varied to ensure the unit conversions of the slopes are correct

    nx, ny = [35, 42]

    hc = 0.2 * sx
    np.random.seed(0)
    topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.,
                                    short_cutoff=hc, long_cutoff=hc+1e-9, )
    topography = topography.scale(1/topography.rms_height())
    interp = topography.interpolate_bicubic()

    fine_topography = topography.interpolate_fourier((4*nx, 4*ny))

    interp_height, interp_slopex, interp_slopey =  interp(*fine_topography.positions(), derivative=1)
    np.testing.assert_allclose(interp_height, fine_topography.heights(), atol=1e-2)
    derx, dery = fine_topography.fourier_derivative()
    rms_slope= topography.rms_slope()
    np.testing.assert_allclose(interp_slopex, derx , atol=1e-1 * rms_slope)
    np.testing.assert_allclose(interp_slopey, dery , atol=1e-1 * rms_slope)


# TODO: check 2nd order derivatives of the wrapper against some function