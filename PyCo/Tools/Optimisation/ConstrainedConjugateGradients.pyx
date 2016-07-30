#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ConstrainedConjugateGradients.pyx

@author Till Junge <till.junge@kit.edu>

@date   08 Dec 2015

@brief  Implements the constrained conjugate gradient algorithm as described in
        I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

@section LICENCE

 Copyright (C) 2015-2016 Till Junge, Lars Pastewka

PyCo is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyCo is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

from libcpp cimport bool
from libc.math cimport sqrt

cdef extern from "cmath" namespace "std":
    bool isnan(double x)

import numpy as np
cimport numpy as cnp
import sys

import scipy.optimize as optim

from cython cimport boundscheck, wraparound, numeric, cdivision
from libc.math cimport fmin, fmax
from cython.parallel cimport prange
from libcpp.string cimport string

# cython: c_string_type=unicode, c_string_encoding=utf8

###
cdef struct MaxCv:
    double max_pen, max_pres

cdef struct cOptimizeResult:
    int nfev, nit
    bint success
    string message
    MaxCv max_cv

cdef dict dict_from_MaxCv(MaxCv m):
    return {'max_pen': m.max_pen,
            'max_pres': m.max_pres}

cdef updateOptimizeResult(result, cOptimizeResult c_result):
    result.nfev = c_result.nfev
    result.nit = c_result.nit
    result.success = c_result.success
    result.message = c_result.message
    result.maxcv = dict_from_MaxCv(c_result.max_cv)


@boundscheck(False)
cpdef constrained_conjugate_gradients(substrate, surface, disp0=None,
                                    double pentol=1e-6, double prestol=1e-5,
                                    int maxiter=100000, logger=None):
    """
    Use a constrained conjugate gradient optimization to find the equilibrium
    configuration deflection of an elastic manifold. The conjugate gradient
    iteration is reset using the steepest descent direction whenever the contact
    area changes.
    Method is described in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

    Parameters
    ----------
    substrate : elastic manifold
        Elastic manifold.
    surface : array_like
        Height profile of the rigid counterbody.
    u_r : array
        Array used for initial displacements. A new array is created if omitted.
    pentol : float
        Maximum penetration of contacting regions required for convergence.
    maxiter : float
        Maximum number of iterations.

    Returns
    -------
    u : array
        2d-array of displacements.
    p : array
        2d-array of pressure.
    converged : bool
        True if iteration stopped due to convergence criterion.
    """

    cdef:
        long A
        int i, j, n_row, n_col, n_row_comp, n_col_comp, nc_r_sum
        double[:, :] disp_view
        double G, G_old, delta, x, tau, max_pres, max_pen
        double[:, ::1] u_r_view, surf_view, p_r_view, g_r_view
        double[:, ::1] t_r_view, r_r_view
        double[:, :] u_r_comp_view, p_r_comp_view, t_r_comp_view
        double[:, :] g_r_comp_view
        cnp.uint8_t[:, ::1] c_r_view
        cnp.uint8_t[:, :] nc_r_comp_view, c_r_comp_view
        cnp.ndarray u_r, surf, t_r, p_r
        cOptimizeResult cresult
        string delta_str
    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    cdef int n_dim = substrate.dim
    if n_dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                 substrate.dim))
    u_r = np.zeros(substrate.computational_resolution)

    if n_dim == 1:
        n_col_comp = n_col = 1

    else:
        n_col = substrate.computational_resolution[1]
        n_col_comp = substrate.resolution[1]
    n_row = substrate.computational_resolution[0]
    n_row_comp = substrate.resolution[0]

    u_r_view = memoryview(u_r)

    if disp0 is not None:
        disp_view = memoryview(disp0)
        for i in prange(n_row, nogil=True):
            for j in range(n_col):
                u_r_view[i, j] = disp_view[i, j]
    else:
        disp_view = memoryview(np.zeros((n_row, n_col)))

    cdef list comp_slice = [slice(0, substrate.resolution[i]) for
                            i in range(substrate.dim)]
    u_r_comp_view = memoryview(u_r[comp_slice])
    surf_view = memoryview(surface)

    for i in prange(n_row_comp, nogil=True):
        for j in range(n_col_comp):
            u_r_comp_view[i, j] = fmax(surf_view[i, j], u_r_comp_view[i, j])
    result = optim.OptimizeResult()

    cresult.nfev = 0
    cresult.nit = 0
    cresult.success = False
    cresult.message = "Not Converged (yet)"

    # Compute forces
    #p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
    p_r = substrate.evaluate_force(u_r).copy()
    p_r_view = memoryview(p_r)
    p_r_comp_view = memoryview(p_r_view[:n_row_comp, :n_col_comp])


    cresult.nfev += 1

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    t_r = np.zeros_like(u_r)
    t_r_view = memoryview(t_r)
    t_r_comp_view = memoryview(t_r_view[:n_row_comp, :n_col_comp])

    c_r_view = memoryview(np.zeros_like(p_r, dtype=np.uint8))
    c_r_comp_view = memoryview(c_r_view[:n_row_comp, :n_col_comp])
    g_r = np.zeros_like(c_r_view, dtype=float)
    g_r_view = memoryview(g_r)
    g_r_comp_view = memoryview(g_r_view[:n_row_comp, :n_col_comp])
    r_r = np.zeros_like(p_r)
    r_r_view = memoryview(r_r)

    nc_r_comp_view = memoryview(np.zeros((n_row_comp, n_col_comp), dtype=np.uint8))

    for it in range(1, maxiter+1):
        cresult.nit = it
        A = 0
        for i in prange(n_row, nogil=True):
            for j in range(n_col):
                # Reset contact area (area that feels compressive stress)
                c_r_view[i, j] = p_r_view[i, j] < 0.
                # Compute total contact area (area with compressive pressure)
                A += c_r_view[i, j]


        # Compute G = sum(g*g) (over contact area only)
        G = 0.
        for i in prange(n_row_comp, nogil=True):
            for j in range(n_col_comp):
                g_r_view[i, j] = u_r_comp_view[i, j] - surf_view[i, j]
                G += c_r_comp_view[i, j] * g_r_view[i, j]*g_r_view[i, j]


        # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
        if delta > 0 and G_old > 0:
            for i in prange(n_row_comp, nogil=True):
                for j in range(n_col_comp):
                    with cdivision(True):
                        t_r_comp_view[i, j] = (
                            c_r_comp_view[i, j] *
                            (g_r_comp_view[i, j] +
                             delta*(G/G_old) * t_r_comp_view[i, j]))
        else:
            for i in prange(n_row_comp, nogil=True):
                for j in range(n_col_comp):
                    t_r_comp_view[i, j] = c_r_comp_view[i, j]*g_r_comp_view[i, j]

        # Compute elastic displacement that belong to t_r
        #substrate (Nelastic manifold: r_r is negative of Polonsky, Kerr's r)
        #r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real
        r_r[...] = substrate.evaluate_disp(t_r)
        cresult.nfev += 1
        # Note: Sign reversed from Polonsky, Keer because this r_r is negative
        # of theirs.
        tau = 0.0
        if A > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = 0.
            for i in prange(n_row, nogil=True):
                for j in range(n_col):
                    x -= c_r_view[i, j] * r_r_view[i, j] * t_r_view[i, j]

            if x > 0.0:
                for i in prange(n_row_comp, nogil=True):
                    for j in range(n_col_comp):
                        with cdivision(True):
                            tau += (c_r_comp_view[i, j] * g_r_comp_view[i, j] *
                                    t_r_comp_view[i,j])/x
            else:
                G = 0.0

        for i in prange(n_row, nogil=True):
            for j in range(n_col):
                p_r_view[i, j] += tau*c_r_view[i, j]*t_r_view[i, j]

        # Find area with tensile stress and negative gap
        # (i.e. penetration of the two surfaces)
        nc_r_sum = 0
        for i in prange(n_row_comp, nogil=True):
            for j in range(n_col_comp):
                nc_r_comp_view[i, j] = (p_r_comp_view[i, j] >= 0.0 and
                                        g_r_comp_view[i, j] <  0.0)
                nc_r_sum += nc_r_comp_view[i, j]

        # Find maximum pressure outside contacting region. This should go to
        # zero.
        # Set all compressive stresses to zero
        max_pres = 0.
        for i in prange(n_row, nogil=True):
            for j in range(n_col):
                #print(p_r_view[i, j]>0.)
                max_pres = fmax(max_pres, (p_r_view[i, j]>0.)*p_r_view[i, j])
                #print(max_pres)
                p_r_view[i, j] *= p_r_view[i, j] < 0.


        if nc_r_sum > 0:
            # nc_r contains area that just jumped into contact. Update their
            # forces.
            for i in prange(n_row_comp, nogil=True):
                for j in range(n_col_comp):
                    p_r_comp_view[i, j] += (tau * nc_r_comp_view[i, j] *
                                            g_r_comp_view[i, j])

            delta = 0
            delta_str = 'sd'
        else:
            delta = 1
            delta_str = 'cg'

        # Compute new displacements from updated forces
        #u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        u_r[...] = memoryview(substrate.evaluate_disp(p_r))
        cresult.nfev += 1


        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A > 0:
            with cdivision(True):
                rms_pen = sqrt(G/A)
        else:
            rms_pen = sqrt(G)
        max_pen = 0.
        for i in prange(n_row_comp, nogil=True):
            for j in range(n_col_comp):
                max_pen = fmax(
                    max_pen,
                    (c_r_comp_view[i, j] *
                     (surf_view[i, j] - u_r_comp_view[i, j])))

        cresult.max_cv.max_pen = max_pen
        cresult.max_cv.max_pres = max_pres

        # Elastic energy would be
        # e_el = -0.5*np.sum(p_r*u_r)

        if rms_pen < pentol and max_pen < pentol and max_pres < prestol:
            if logger is not None:
                logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen',
                           'max_pres'],
                          ['CONVERGED', it, A, tau, rms_pen, max_pen, max_pres],
                          force_print=True)
            result.x = u_r#[comp_slice]
            result.jac = -p_r[comp_slice]
            cresult.success = True
            cresult.message = "Polonsky converged"
            updateOptimizeResult(result, cresult)
            return result

        if logger is not None:
            logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen',
                       'max_pres'],
                      [delta_str, it, A, tau, rms_pen, max_pen, max_pres])

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    result.x = u_r.copy()#[comp_slice]
    result.jac = -p_r[comp_slice]
    print(("###########################################\n"
           "##### jac.sum = {}\n"
           "##### x.sum = {}\n"
           "##### u_r.sum = {}\n"
           "###########################################"
           "\n").format(result.x.sum(), result.jac.sum(), u_r.sum()))
    cresult.message = "Reached maxiter = {}".format(maxiter).encode('latin-1')
    updateOptimizeResult(result, cresult)
    return result
