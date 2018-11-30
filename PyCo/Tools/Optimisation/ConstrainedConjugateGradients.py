#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   ConstrainedConjugateGradients.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   30 Jul 2016

@brief  Dispatcher for the constrained conjugate gradient algorithm as described
        in I.A. Polonsky, L.M. Keer, Wear 231, 206 (1999)

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

from .ConstrainedConjugateGradientsPy import constrained_conjugate_gradients as ref

# FIXME: This is a convenience fix because I get following error
#        when using mpirun
#        ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /home/fr/fr_fr/fr_as1412/development_debug/PyCo/PyCo/Tools/Optimisation/ConstrainedConjugateGradientsOpt.cpython-36m-x86_64-linux-gnu.so)

try :
    from .ConstrainedConjugateGradientsOpt import constrained_conjugate_gradients as opt
except ImportError:
    print("ConstrainedConjugateGradientsOpt not available, don't use ")
    opt = None

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
