#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ConstrainedConjugateGradients.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   30 Jul 2016

@brief  Dispatcher for the constrained conjugate gradient algorithm as described
        in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

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

from .ConstrainedConjugateGradientsPy import constrained_conjugate_gradients as ref
from .ConstrainedConjugateGradientsOpt import constrained_conjugate_gradients as opt

def constrained_conjugate_gradients(substrate, surface, kind='ref', **kwargs):
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
    kind : str
        Decide which solver to call. 'ref' is pure Python reference
        implementation, 'opt' is optimized and parallelized Cython
        implementation (that requires continuous buffers).

    Returns
    -------
    u : array
        2d-array of displacements.
    p : array
        2d-array of pressure.
    converged : bool
        True if iteration stopped due to convergence criterion.
    """

    solvers = dict(ref=ref, opt=opt)

    if kind not in solvers:
        raise ValueError("Unknown solver kind '{}'.".format(kind))
    return solvers[kind](substrate, surface, **kwargs)
