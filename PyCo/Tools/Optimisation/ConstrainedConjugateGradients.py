#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ConstrainedConjugateGradients.py

@author Lars Pastewka <lars.pastewka@kit.edu>

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

from math import isnan, pi, sqrt

import numpy as np

import scipy.optimize as optim

###

def constrained_conjugate_gradients(substrate, surface, disp0=None, pentol=1e-6,
                                    prestol=1e-5, maxiter=100000, logger=None):
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

    # Note: Suffix _r deontes real-space _q reciprocal space 2d-arrays

    if logger is not None:
        logger.pr('maxiter = {0}'.format(maxiter))
        logger.pr('pentol = {0}'.format(pentol))

    if disp0 is None:
        u_r = np.zeros(substrate.computational_resolution)
    else:
        u_r = disp0.copy()


    comp_slice = [slice(0, substrate.resolution[i]) for i in range(substrate.dim)]
    if substrate.dim not in (1, 2):
        raise Exception(
            ("Constrained conjugate gradient currently only implemented for 1 "
             "or 2 dimensions (Your substrate has {}.).").format(
                 substrate.dim))
    u_r[comp_slice] = np.where(u_r[comp_slice] < surface, surface, u_r[comp_slice])

    result = optim.OptimizeResult()
    result.nfev = 0
    result.nit = 0
    result.success = False
    result.message = "Not Converged (yet)"

    # Compute forces
    #p_r = -np.fft.ifft2(np.fft.fft2(u_r)/gf_q).real
    p_r = substrate.evaluate_force(u_r)
    result.nfev += 1

    # iteration
    delta = 0
    delta_str = 'reset'
    G_old = 1.0
    t_r = np.zeros_like(u_r)


    for it in range(1, maxiter+1):
        result.nit = it
        # Reset contact area (area that feels compressive stress)
        c_r = p_r < 0.0

        # Compute total contact area (area with compressive pressure)
        A = np.sum(c_r)

        # Compute G = sum(g*g) (over contact area only)
        g_r = u_r[comp_slice]-surface
        G = np.sum(c_r[comp_slice]*g_r*g_r)

        # t = (g + delta*(G/G_old)*t) inside contact area and 0 outside
        if delta > 0 and G_old > 0:
            t_r[comp_slice] = c_r[comp_slice]*(g_r + delta*(G/G_old)*t_r[comp_slice])
        else:
            t_r[comp_slice] = c_r[comp_slice]*g_r

        # Compute elastic displacement that belong to t_r
        #substrate (Nelastic manifold: r_r is negative of Polonsky, Kerr's r)
        #r_r = -np.fft.ifft2(gf_q*np.fft.fft2(t_r)).real
        r_r = substrate.evaluate_disp(t_r)
        result.nfev += 1
        # Note: Sign reversed from Polonsky, Keer because this r_r is negative
        # of theirs.
        tau = 0.0
        if A > 0:
            # tau = -sum(g*t)/sum(r*t) where sum is only over contact region
            x = -np.sum(c_r*r_r*t_r)
            if x > 0.0:
                tau = np.sum(c_r[comp_slice]*g_r*t_r[comp_slice])/x
            else:
                G = 0.0

        p_r += tau*c_r*t_r

        # Find area with tensile stress and negative gap
        # (i.e. penetration of the two surfaces)
        nc_r = np.logical_and(p_r[comp_slice] >= 0.0, g_r < 0.0)

        # Find maximum pressure outside contacting region. This should go to
        # zero.
        max_pres = 0
        mask = p_r>0
        if mask.sum() > 0:
            max_pres = p_r[mask].max()

        # Set all compressive stresses to zero
        p_r *= p_r < 0.0

        if np.sum(nc_r) > 0:
            # nc_r contains area that just jumped into contact. Update their
            # forces.
            p_r[comp_slice] += tau*nc_r*g_r

            delta = 0
            delta_str = 'sd'
        else:
            delta = 1
            delta_str = 'cg'

        # Compute new displacements from updated forces
        #u_r = -np.fft.ifft2(gf_q*np.fft.fft2(p_r)).real
        u_r = substrate.evaluate_disp(p_r)
        result.nfev += 1


        # Store G for next step
        G_old = G

        # Compute root-mean square penetration, max penetration and max force
        # difference between the steps
        if A > 0:
            rms_pen = sqrt(G/A)
        else:
            rms_pen = sqrt(G)
        max_pen = max(0.0, np.max(c_r[comp_slice]*(surface-u_r[comp_slice])))
        result.maxcv = {"max_pen": max_pen,
                        "max_pres": max_pres}

        # Elastic energy would be
        # e_el = -0.5*np.sum(p_r*u_r)

        if rms_pen < pentol and max_pen < pentol and max_pres < prestol:
            if logger is not None:
                logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen',
                           'max_pres'],
                          ['CONVERGED', it, A, tau, rms_pen, max_pen, max_pres],
                          force_print=True)
            result.x = u_r[comp_slice]
            result.jac = -p_r[comp_slice]
            result.success = True
            result.message = "Polonsky converged"
            return result

        if logger is not None:
            logger.st(['status', 'it', 'A', 'tau', 'rms_pen', 'max_pen',
                       'max_pres'],
                      [delta_str, it, A, tau, rms_pen, max_pen, max_pres])

        if isnan(G) or isnan(rms_pen):
            raise RuntimeError('nan encountered.')

    result.message = "Reached maxiter = {}".format(maxiter)
    return result
