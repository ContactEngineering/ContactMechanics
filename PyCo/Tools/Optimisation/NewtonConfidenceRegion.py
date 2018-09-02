#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   NewtonConfidenceRegion.py

@author Till Junge <till.junge@kit.edu>

@date   17 Sep 2015

@brief  implements the Newton method with confidence region as defined in
        Bierlaire (2006):
        Introduction à l'optimization différentiable, Michel Bierlaire,
        Presses polytechniques et universitaires romandes, Lausanne 2006,
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
from .common import steihaug_toint


# Algorithm 12.4 (p.301) Newton method with confidence interval
def newton_confidence_region(fun, x0, jac, hess, tol, store_iterates=None,
                             radius0=10., eta1=0.01, eta2=.9,
                             method=steihaug_toint, **options):
    """

    Keyword Arguments:
    fun            -- objective function to minimize
    x0             -- initial guess for solution
    jac            -- Jacobian (gradient) of objective function
    hess           -- Hessian (matrix of second-order derivatives) of objective
                      function
    tol            -- Tolerance for termination
    store_iterates -- (default None) if set to 'iterate' the full iterates are
                      stored in module-level constant iterates
    radius0        -- (default 10) size of initial confidence region size
    eta1/eta2      -- (default 0.01, 0.9) heuristics for step length
                      modifications. Defaults from Bierlaire (2006)
    method         -- (default 'steihaug_toint') solver for confidence region
                      sub-problem. can be either steihaug_toint or dogleg
    **options      -- none of those will be used
    """

    # initialisation
    maxiter_key = 'maxiter'
    if maxiter_key not in options.keys():
        options[maxiter_key] = 20

    counter = 0
    x = np.array(x0.copy()).reshape((-1, 1))
    radius = radius0
    iterates = list()
    state = ''
    if store_iterates == 'iterate':
        iterate = scipy.optimize.OptimizeResult(
            {'x': x.copy(),
             'fun': fun(x),
             'jac': jac(x),
             'hess': hess(x),
             'radius': radius,
             'rho': float('nan'),
             'state': state})
        iterates.append(iterate)
    try:
        while True:
            b = np.array(jac(x))  # pylint: disable=invalid-name
            norm_grad = np.linalg.norm(b)
            if norm_grad < tol:
                raise ReachedTolerance(
                    "||grad f(x)|| = {} < {} = tol".format(
                        norm_grad, tol))
            if counter == options['maxiter']:
                raise ReachedMaxiter("reached maxiter ({})".format(
                    options['maxiter']))

            # 1) solve sub-problem
            Q = np.array(hess(x))  # pylint: disable=invalid-name
            direction = method(b, Q, radius)

            # 2)
            xpd = x + direction
            rho = ((fun(x)-fun(xpd)) /
                   (-float(.5*direction.T@Q@direction + direction.T@b)))

            # 3)
            if rho < eta1:
                # fail
                state = '-'
                radius = .5 * np.linalg.norm(direction)
            else:
                x += direction
                state = '+'
                if rho > eta2:
                    # very good progress
                    state = '++'
                    radius *= 2
            counter += 1

            if store_iterates == 'iterate':
                iterate = scipy.optimize.OptimizeResult(
                    {'x': x.copy(),
                     'fun': fun(x),
                     'jac': jac(x),
                     'hess': hess(x),
                     'radius': radius,
                     'rho': rho,
                     'state': state})
                iterates.append(iterate)

    except ReachedMaxiter as err:
        message = str(err)
        success = False
    except(ReachedTolerance) as err:
        message = str(err)
        success = True

    result = scipy.optimize.OptimizeResult({'message': message,
                                            'success': success,
                                            'x': x,
                                            'fun': fun(x),
                                            'jac': jac(x),
                                            'hess': hess(x),
                                            'nit': counter})

    if iterates:
        result['iterates'] = iterates
    return result
