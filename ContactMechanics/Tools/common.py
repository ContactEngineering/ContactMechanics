#
# Copyright 2016, 2018, 2020 Lars Pastewka
#           2019 Antoine Sanner
#           2015-2016 Till Junge
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
Bin for small common helper function and classes
"""

import numpy as np


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
        x_plus[i] += .5 * delta
        x_minus = x.copy().reshape(-1)
        x_minus[i] -= .5 * delta
        grad[i] = (fun(x_plus.reshape(x.shape)) -
                   fun(x_minus.reshape(x.shape))) / delta
    grad.shape = x.shape
    return grad


def mean_err(arr1, arr2, rfft=False):
    "computes the mean element-wise difference between two containers"
    comp_sl = [slice(0, max(d_1, d_2)) for (d_1, d_2)
               in zip(arr1.shape, arr2.shape)]
    if rfft:
        comp_sl[-1] = slice(0, comp_sl[-1].stop // 2 + 1)
    if arr1[tuple(comp_sl)].shape != arr2[tuple(comp_sl)].shape:
        raise Exception("The array shapes differ: a: {}, b:{}".format(
            arr1.shape, arr2.shape))
    return abs(np.ravel(arr1[tuple(comp_sl)] - arr2[tuple(comp_sl)])).mean()


def compute_wavevectors(nb_grid_pts, physical_sizes, nb_dims):
    """
    computes and returns the wavevectors q that exist for the surfaces
    physical_sizes and nb_grid_pts as one vector of components per dimension
    """
    vectors = list()
    if nb_dims == 1:
        nb_grid_pts = [nb_grid_pts]
        physical_sizes = [physical_sizes]
    for dim in range(nb_dims):
        vectors.append(2 * np.pi * np.fft.fftfreq(
            nb_grid_pts[dim],
            physical_sizes[dim] / nb_grid_pts[dim]))
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
    return integral / np.prod(arr.shape) * np.fft.fftn(arr)


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
    return np.prod(arr.shape) / integral * np.fft.ifftn(arr)


def get_q_from_lambda(lambda_min, lambda_max):
    """ Conversion between wavelength and angular frequency
    """
    if lambda_min == 0:
        q_max = float('inf')
    else:
        q_max = 2 * np.pi / lambda_min
    q_min = 2 * np.pi / lambda_max
    return q_min, q_max
