#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   common.py

@author Till Junge <till.junge@kit.edu>

@date   11 Feb 2015

@brief  Bin for small common helper function and classes

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

from .common import _get_size, radial_average


def autocorrelation_1D(surface_xy,  # pylint: disable=invalid-name
                       size=None, periodic=False):
    """
    Compute height-difference autocorrelation function and radial average.

    Parameters
    ----------
    surface_xy : array_like
        2D-array of surface topography
    size : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of grid
        points.)
    periodic : bool, optional
        Topography is periodic topography map. (Default: False)

    Returns
    -------
    r : array
        Distances. (Units: length)
    A : array
        Autocorrelation function. (Units: length**2)
    """
    nx, dummy_ny = surface_xy.shape
    sx, dummy_sy = _get_size(surface_xy, size)

    # Compute FFT and normalize
    if periodic:
        surface_qy = np.fft.fft(surface_xy[:, :], axis=0)
        C_qy = abs(surface_qy)**2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft(C_qy, axis=0).real/nx

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = A_xy[0, :]-A_xy

        A = A_xy[:nx//2, :]
        A[1:nx//2, :] += A_xy[nx-1:(nx+1)//2:-1, :]
        A /= 2

        r = sx*np.arange(nx//2)/nx
    else:
        surface_qy = np.fft.fft(surface_xy[:, :], n=2*nx-1, axis=0)
        C_qy = abs(surface_qy)**2  # pylint: disable=invalid-name
        normx = (np.abs(np.arange(2*nx-1) - (nx-1))+1).reshape(-1, 1)
        A_xy = np.fft.ifft(C_qy, axis=0).real/normx

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = A_xy[0, :]-A_xy

        A = A_xy[:nx, :]
        A[1:nx, :] += A_xy[2*nx-1:nx-1:-1, :]
        A /= 2

        r = sx*np.arange(nx)/nx

    return r, A.mean(axis=1)

def autocorrelation_2D(surface_xy, nbins=100,  # pylint: disable=invalid-name
                       size=None, periodic=False, return_map=False):
    """
    Compute height-difference autocorrelation function and radial average.

    Parameters
    ----------
    surface_xy : array_like
        2D-array of surface topography
    nbins : int
        Number of bins for radial average. Note: Returned array can be smaller
        than this because bins without data point are discarded.
    size : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of grid
        points.)
    periodic : bool, optional
        Topography is periodic topography map. (Default: False)
    return_map : bool, optional
        Return full 2D autocorrelation map. (Default: False)

    Returns
    -------
    r : array
        Distances. (Units: length)
    A : array
        Autocorrelation function. (Units: length**2)
    A_xy : array
        2D autocorrelation function. Only returned if return_map=True.
        (Units: length**2)
    """
    nx, ny = surface_xy.shape
    sx, sy = _get_size(surface_xy, size)

    # Pixel size
    area0 = (sx/nx)*(sy/ny)

    # Compute FFT and normalize
    if periodic:
        surface_qk = np.fft.fft2(surface_xy[:, :])
        C_qk = abs(surface_qk)**2  # pylint: disable=invalid-name
        A_xy = np.fft.ifft2(C_qk).real/(nx*ny)

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = A_xy[0, 0]-A_xy

        if nbins is None:
            return A_xy

        # Radial average
        r_edges, n, r_val, A_val = radial_average(  # pylint: disable=invalid-name
            A_xy, (sx+sy)/4, nbins, size=(sx, sy))
    else:
        surface_qk = np.fft.fft2(surface_xy[:, :], s=(2*nx-1, 2*ny-1))
        C_qk = abs(surface_qk)**2  # pylint: disable=invalid-name
        normx = (np.abs(np.arange(2*nx-1) - (nx-1))+1).reshape(-1, 1)
        normy = (np.abs(np.arange(2*ny-1) - (ny-1))+1).reshape(1, -1)
        A_xy = np.fft.ifft2(C_qk).real/(normx*normy)

        # Convert height-height autocorrelation to height-difference
        # autocorrelation
        A_xy = A_xy[0, 0]-A_xy

        if nbins is None:
            return A_xy

        # Radial average
        r_edges, n, r_val, A_val = radial_average(  # pylint: disable=invalid-name
            A_xy, (sx+sy)/2, nbins, size=(sx*(2*nx-1)/nx, sy*(2*ny-1)/ny))

    if return_map:
        return r_val[n>0], A_val[n>0], A_xy
    else:
        return r_val[n>0], A_val[n>0]
