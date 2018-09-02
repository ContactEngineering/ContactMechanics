#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   common.py

@author Till Junge <till.junge@kit.edu>

@date   11 Feb 2015

@brief  Bin for small common helper function and classes

@section LICENCE

Copyright 2015-2018 Till Junge, Lars Pastewka

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
import scipy


def compare_containers(cont_a, cont_b):
    """
    compares whether two containers have the same content regardless of their
    type. eg, compares [1, 2, 3] and (1, 2., 3.) as True
    Keyword Arguments:
    cont_a -- one container
    cont_b -- other container
    """
    if cont_a != cont_b:
        # pylint: disable=broad-except
        # pylint: disable=multiple-statements
        try:
            if not len(cont_a) == len(cont_b):
                return False
            for a_item, b_item in zip(cont_a, cont_b):
                if not a_item == b_item:
                    return False
        except Exception:
            return False
    return True


def evaluate_gradient(fun, x, delta):
    """
    Don't actually use this function in production code, it's not efficient.
    Returns an approximation of the gradient evaluated using central
    differences.

    (grad f)_i = ((f(x+e_i*delta/2)-f(x-e_i*delta/2))/delta)

    Arguments:
    fun   -- function to be evaluated
    x     -- point of evaluation
    delta -- step width
    """
    # pylint: disable=invalid-name
    x = np.array(x)
    grad = np.zeros_like(x).reshape(-1)
    for i in range(x.size):
        # pylint: disable=bad-whitespace
        x_plus = x.copy().reshape(-1)
        x_plus[i] += .5*delta
        x_minus = x.copy().reshape(-1)
        x_minus[i] -= .5*delta
        grad[i] = (fun(x_plus.reshape(x.shape)) -
                   fun(x_minus.reshape(x.shape)))/delta
    grad.shape = x.shape
    return grad


def mean_err(arr1, arr2, rfft=False):
    "computes the mean element-wise difference between two containers"
    comp_sl = [slice(0, max(d_1, d_2)) for (d_1, d_2)
               in zip(arr1.shape, arr2.shape)]
    if rfft:
        comp_sl[-1] = slice(0, comp_sl[-1].stop//2+1)
    if arr1[tuple(comp_sl)].shape != arr2[tuple(comp_sl)].shape:
        raise Exception("The array shapes differ: a: {}, b:{}".format(
            arr1.shape, arr2.shape))
    return abs(np.ravel(arr1[tuple(comp_sl)]-arr2[tuple(comp_sl)])).mean()


def compute_wavevectors(resolution, size, nb_dims):
    """
    computes and returns the wavevectors q that exist for the surfaces size
    and resolution as one vector of components per dimension
    """
    vectors = list()
    if nb_dims == 1:
        resolution = [resolution]
        size = [size]
    for dim in range(nb_dims):
        vectors.append(2*np.pi*np.fft.fftfreq(
            resolution[dim],
            size[dim]/resolution[dim]))
    return vectors


def fftn(arr, integral):
    """
    n-dimensional fft according to the conventions detailed in
    power_spectrum.tex in the notes folder.

    Keyword Arguments:
    arr      -- Input array, can be complex
    integral -- depending of dimensionality:
                1D: Length of domain
                2D: Area
                etc
    """
    return integral/np.prod(arr.shape)*np.fft.fftn(arr)


def ifftn(arr, integral):
    """
    n-dimensional ifft according to the conventions detailed in
    power_spectrum.tex in the notes folder.

    Keyword Arguments:
    arr      -- Input array, can be complex
    integral -- depending of dimensionality:
                1D: Length of domain
                2D: Area
                etc
    """
    return np.prod(arr.shape)/integral*np.fft.ifftn(arr)


def _get_size(surface_xy, size=None):
    """
    Get the physical size of the topography map. Defaults to the shape of
    the array if no other information is present.
    """
    if size is None:
        if isinstance(surface_xy, np.ndarray):
            size = surface_xy.shape
        else:
            try:
                size = surface_xy.size
            except:
                pass
    if size is None:
        size = surface_xy.shape
    return size


def compute_derivative(profile, size=None, dim=None, n=1):
    """
    Compute local slope
    """
    resolution = profile.shape
    size = _get_size(profile, size)

    grid_spacing = np.array(size)/np.array(resolution)
    if dim is None:
        dims = range(len(profile.shape))
    else:
        dims = range(dim)
    return [np.diff(profile[...], n=n, axis=d)/grid_spacing[d]**n
            for d in dims]


def compute_tilt_from_height(arr, size=None, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()
    """
    size = _get_size(arr, size)
    arr = arr[...]
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    #res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)
    coeffs = np.array(res[0])*\
        np.array(list(arr.shape)+[1.])/np.array(list(size)+[1.])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def compute_tilt_from_slope(arr, size=None):
    return [x.mean() for x in compute_derivative(arr, size)]


def compute_tilt_and_curvature(arr, size=None, full_output=False):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface.

    idea as follows

    1) arr = arr_out + (ň.x + d)/ň_z
    2) arr_out.sum() = 0
    3) |ň| = 1
    => n_z = sqrt(1 - n_x^2 - n_y^2) (for 2D, but you get the idea)
       dofs = n_x, n_y, d = X

    solution X_s = arg_min ((arr - ň.x + d)^2).sum()
    """
    size = _get_size(arr, size)
    arr = arr[...]
    nb_dim = len(arr.shape)
    assert nb_dim == 2
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    # Linear terms
    x_grids = np.meshgrid(*x_grids, indexing='ij')
    # Quadratic terms
    x, y = x_grids
    x_grids += [x*x, y*y, x*y]
    if np.ma.getmask(arr) is np.ma.nomask:
        columns = [x.reshape((-1, 1)) for x in x_grids]
    else:
        columns = [x[np.logical_not(arr.mask)].reshape((-1, 1))
                   for x in x_grids]
    columns.append(np.ones_like(columns[-1]))
    # linear regression model
    location_matrix = np.hstack(columns)
    offsets = np.ma.compressed(arr)
    #res = scipy.optimize.nnls(location_matrix, offsets)
    res = np.linalg.lstsq(location_matrix, offsets, rcond=None)

    nx, ny = arr.shape
    sx, sy = size

    x, y, xx, yy, xy, z = res[0]
    coeffs = np.array([x*nx/sx, y*ny/sy, xx*(nx/sx)**2, yy*(ny/sy)**2,
                       xy*nx/sx*ny/sy, z])
    if full_output:
        return coeffs, location_matrix
    else:
        return coeffs


def shift_and_tilt(arr, full_output=False):
    """
    returns an array of same shape and size as arr, but shifted and tilted so
    that mean(arr) = 0 and mean(arr**2) is minimized
    """
    coeffs, location_matrix = compute_tilt_from_height(arr, full_output=True)
    coeffs = np.matrix(coeffs).T
    offsets = np.matrix(arr[...].reshape((-1, 1)))
    if full_output:
        return ((offsets-location_matrix*coeffs).reshape(arr.shape),
                coeffs, res[1])
    else:
        return (offsets-location_matrix*coeffs).reshape(arr.shape)


def shift_and_tilt_approx(arr, full_output=False):
    """
    does the same as shift_and_tilt, but computes an iterative approximation.
    Use in case of large surfaces.
    """
    nb_dim = len(arr.shape)
    x_grids = (np.arange(arr.shape[i]) for i in range(nb_dim))
    if nb_dim > 1:
        x_grids = np.meshgrid(*x_grids, indexing='ij')
    if nb_dim == 2:
        sx_ = x_grids[0].sum()
        sy_ = x_grids[1].sum()
        s__ = np.prod(arr.shape)
        sxx = (x_grids[0]**2).sum()
        sxy = (x_grids[0]*x_grids[1]).sum()
        syy = (x_grids[1]**2).sum()
        sh_ = arr.sum()
        shx = (arr*x_grids[0]).sum()
        shy = (arr*x_grids[1]).sum()
        location_matrix = np.array(((sxx, sxy, sx_),
                                    (sxy, syy, sy_),
                                    (sx_, sy_, s__)))
        offsets = np.array(((shx,),
                            (shy,),
                            (sh_, )))
        coeffs = scipy.linalg.solve(location_matrix, offsets)
        corrective = coeffs[0]*x_grids[0] + coeffs[1]*x_grids[1] + coeffs[2]
        if full_output:
            return arr - corrective, coeffs
        else:
            return arr - corrective


def shift_and_tilt_from_slope(arr, size=None):
    """
    Data in arr is interpreted as height information of a tilted and shifted
    surface. returns an array of same shape and size, but shifted and tilted so
    that mean(arr) = 0 and mean(arr') = 0
    """
    nx, ny = arr.shape
    mean_slope = compute_tilt_from_slope(arr, size)
    tilt_correction = sum([x*y for x, y in
                           zip(mean_slope,
                               np.meshgrid(np.arange(nx)-nx//2,
                                           np.arange(ny)-ny//2,
                                           indexing='ij'))])
    arr = arr - tilt_correction
    return arr - arr.mean()


def radial_average(C_xy, rmax, nbins, size=None):
    """
    Compute radial average of quantities reported on a 2D grid.

    Parameters
    ----------
    C_xy : array_like
        2D-array of values to be averaged.
    rmax : float
        Maximum radius.
    nbins : int
        Number of bins for averaging.
    size : (float, float), optional
        Physical size of the 2D grid. (Default: Size is equal to number of grid
        points.)

    Returns
    -------
    r_edges : array
        Bin edges.
    r_averages : array
        Bin centers, obtained by averaging actual distance values.
    n : array
        Number of data points per radial grid.
    C_r : array
        Averaged values.
    """
    # pylint: disable=invalid-name
    nx, ny = C_xy.shape
    sx = sy = 1.
    x = np.arange(nx)
    x = np.where(x > nx//2, nx-x, x)/nx
    y = np.arange(ny)
    y = np.where(y > ny//2, ny-y, y)/ny

    rmin = 0.0

    if size is not None:
        sx, sy = size
        x *= sx
        y *= sy
        rmin = min(sx/nx, sy/ny)
    dr_xy = np.sqrt((x**2).reshape(-1, 1) + (y**2).reshape(1, -1))

    # Quadratic -> similar statistics for each data point
    # dr_r        = np.sqrt( np.linspace(0, rmax**2, nbins) )

    # Power law -> equally spaced on a log-log plot
    dr_r = np.exp(np.linspace(np.log(rmin), np.log(rmax), nbins))

    dr_max = np.max(dr_xy)
    # Keep dr_max sorted
    if dr_max > dr_r[-1]:
        dr_r = np.append(dr_r, [dr_max+0.1])
    else:
        dr_r = np.append(dr_r, [dr_r[-1]+0.1])

    # Linear interpolation
    dr_xy = np.ravel(dr_xy)
    C_xy = np.ravel(C_xy)
    i_xy = np.searchsorted(dr_r, dr_xy)

    n_r = np.bincount(i_xy, minlength=len(dr_r))
    dravg_r = np.bincount(i_xy, weights=dr_xy, minlength=len(dr_r))
    C_r = np.bincount(i_xy, weights=C_xy, minlength=len(dr_r))

    nreg_r = np.where(n_r == 0, np.ones_like(n_r), n_r)
    dravg_r /= nreg_r
    C_r /= nreg_r

    return np.append([0.0], dr_r), n_r, dravg_r, C_r


def compute_rms_height(profile, kind='Sq'):
    "computes the rms height fluctuation of the surface"
    if kind == 'Sq':
        return np.sqrt(((profile[...]-profile[...].mean())**2).mean())
    elif kind == 'Rq':
        return np.sqrt(((profile[...]-profile[...].mean(axis=0))**2).mean())
    else:
        raise RuntimeError("Unknown rms height kind '{}'.".format(kind))


def compute_rms_slope(profile, size=None, dim=None):
    "computes the rms height gradient fluctuation of the surface"
    diff = compute_derivative(profile, size, dim)
    return np.sqrt((diff[0]**2).mean()+(diff[1]**2).mean())


def compute_rms_curvature(profile, size=None, dim=None):
    "computes the rms height gradient fluctuation of the surface"
    curv = compute_derivative(profile, size, dim, n=2)
    return np.sqrt(((curv[0][:, 1:-1]+curv[1][1:-1, :])**2).mean())


def get_q_from_lambda(lambda_min, lambda_max):
    """ Conversion between wavelength and angular frequency
    """
    if lambda_min == 0:
        q_max = float('inf')
    else:
        q_max = 2*np.pi/lambda_min
    q_min = 2*np.pi/lambda_max
    return q_min, q_max
