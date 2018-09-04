#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   NewtonLineSearch.py

@author Till Junge <till.junge@kit.edu>

@date   16 Sep 2015

@brief  Implements the Newton method with line search as defined in
        Bierlaire (2006):
        Introduction à l'optimization différentiable, Michel Bierlaire, Presses
        polytechniques et universitaires romandes, Lausanne 2006,
        ISBN:2-88074-669-8

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
import scipy
import scipy.optimize

from .common import ReachedTolerance, ReachedMaxiter
from .common import modified_cholesky, line_search


# -----------------------------------------------------------------------------
# implemented as a custom minimizer for scipy
def newton_linesearch(fun, x0, jac, hess, tol, args=(), store_iterates=None,
                      **options):
    """
    see Bierlaire (2006), p. 278
    Keyword Arguments:
    fun       -- objective function to minimize
    x0        -- initial guess for solution
    jac       -- Jacobian (gradient) of objective function
    hess      -- Hessian (matrix of second-order derivatives) of objective
                 function
    tol       -- Tolerance for termination
    store_iterates -- (default None) if set to 'iterate' the full iterates are
                   stored in module-level constant iterates
    **options -- none of those will be used
    """
    # pylint: disable=too-many-branches
    x = np.array(x0.copy()).reshape((-1, 1))
    try:
        fprime = jac(x, *args)
    except Exception:
        print(jac, type(jac))
        raise

    maxiter_key = 'maxiter'
    if maxiter_key not in options.keys():
        options[maxiter_key] = 20

    linesearch_maxiter_key = 'linesearch_maxiter'

    if linesearch_maxiter_key not in options.keys():
        options[linesearch_maxiter_key] = 20

    counter = 0
    iterates = list()
    if store_iterates == 'iterate':
        iterate = scipy.optimize.OptimizeResult(
            {'x': x.copy(),
             'fun': fun(x, *args),
             'jac': jac(x, *args),
             'hess': hess(x, *args),
             'tau': float('nan'),
             'alpha': float('nan')})
        iterates.append(iterate)
    if args:
        def use_fun(x):  # pylint: disable=missing-docstring
            return fun(x, *args)

        def use_jac(x):  # pylint: disable=missing-docstring
            return jac(x, *args)

        def use_hess(x):  # pylint: disable=missing-docstring
            return hess(x, *args)
    else:
        use_fun = fun
        use_jac = jac
        use_hess = hess
    try:
        while True:
            try:
                norm_grad = np.linalg.norm(fprime)
            except Exception:
                print(fprime)
                print(type(fprime))
                print(fprime.dtype)
                print(fprime.shape)
                raise
            if norm_grad < tol:
                raise ReachedTolerance(
                    "||grad f(x)|| = {} < {} = tol".format(
                        norm_grad, tol))
            if counter == options['maxiter']:
                raise ReachedMaxiter("reached maxiter ({})".format(
                    options['maxiter']))
            # 1)
            L, tau = modified_cholesky(  # pylint: disable=invalid-name
                hess(x, *args))
            # 2)
            fprime = use_jac(x)
            z = np.linalg.solve(L, fprime)  # pylint: disable=invalid-name
            # 3)
            d = np.linalg.solve(L.T, -z)  # pylint: disable=invalid-name
            # 4)
            result = line_search(use_fun, x, use_jac, d, alpha0=1,
                                 maxiter=options[linesearch_maxiter_key])
            alpha = result.x
            # violation = result.violation

            # 5)
            x += alpha * d
            counter += 1

            if store_iterates == 'iterate':
                iterate = scipy.optimize.OptimizeResult(
                    {'x': x.copy(),
                     'fun': use_fun(x),
                     'jac': use_jac(x),
                     'hess': use_hess(x),
                     'tau': tau,
                     'alpha': alpha})
                iterates.append(iterate)

    except ReachedMaxiter as err:
        message = str(err)
        success = False
    except(ReachedTolerance) as err:
        message = str(err)
        success = True

    result = scipy.optimize.OptimizeResult({'message': message,
                                            'success': success,
                                            'x': np.asarray(x).ravel(),
                                            'fun': use_fun(x),
                                            'jac': use_jac(x),
                                            'hess': use_hess(x),
                                            'nit': counter})

    if iterates:
        result['iterates'] = iterates
    return result
