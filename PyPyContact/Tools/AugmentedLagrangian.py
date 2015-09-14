#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   AugmentedLagrangian.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Implements the AugmentedLagrangian as described in
        Nocedal, Jorge; Wright, Stephen J. (2006), Numerical Optimization
        (2nd ed.), Berlin, New York: Springer-Verlag, ISBN 978-0-387-30303-1

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""
import numpy as np
import scipy

# implemented as a custom minimizer for scipy
def augmented_lagrangian(fun, x0, args=(), constraints=None, tol=None,
                         update_tol0=.1, multiplier0=0, penalty0=10, alpha=0.1,
                         beta=0.9, tau=10, min_method='L-BFGS-B', **options):
    """
    Custom minimizer that implements the LANCELOT (Conn et al., 1992) augmented
    Lagrangian minimizer. For documentation, see Bierlaire (2006)


    Bierlaire (2006):
    Introduction à l'optimization différentiable, Michel Bierlaire, Presses
    polytechniques et universitaires romandes, Lausanne 2006,
    ISBN:2-88074-669-8

    Keyword Arguments:
    fun         -- objective function to minimize
    x0          -- initial guess for solution
    args        -- (default empty) additional arguments that need to be fed to fun
    constraints -- (default None) not optional. An exception will be raised if
                   this is not specified. A callable that is called as
                   constraints(x, args=args) and returns a ndarray with the
                   same shape as the return-vector of fun
    tol         -- (default None) Convergence tolerance. Is used for both the
                   subjacent minimizer as well as for the outer (augmented
                   Lagrangian) loop.
    update_tol0 -- (default .1) This value is used to decide whether the
                   minimum x_k is "sufficiently" admissible: if
                   ||constraints(x_k, args=args)|| < current update_tol, then
                   the multipliers are updated, else the penalty is increased.
    multiplier0 -- (default 0) Initial guess for the Lagrange multipliers
    penalty0    -- (default 10) Initial penalty value (default from LANCELOT)
    alpha       -- (default 0.1) defines how aggressively the update_tol is
                   reset everytime the penalty is increased. This is a
                   heuristic default from LANCELOT.
    beta        -- (default 0.9) defines how aggressively the update_tol is
                   updated everytime the Lagrange multipliers are updated. This
                   is a heuristic default from LANCELOT.
    tau         -- (default 10) defines how aggressively to update the penalty.
                   This is a heuristic default from LANCELOT.
    min_method  -- (default 'L-BFGS-B') see scipy documentation for details. If
                   you change this, be sure to choose method that can handle
                   high-dimensional parameter spaces and that does not
                   interfere with the augmented Lagrangian algorithm (i.e., if
                   the method you choose handles the constraints, chances are
                   that the algo will not play nice with it). Is handed as
                   'method' keyword parameter to scipy.optimize.minimize.
    **options   -- are handed through to min_method as is.
    """
    x = x0
    multiplier = multiplier0  # 'lam' in the objective
    penalty = penalty0        # 'c_pen' in the objective
    update_tol0 = penalty**alpha*update_tol0
    update_tol = update_tol0/penalty**alpha
    current_tol = tol

    def mod_objective_no_args(x, lam, c_pen):
        """ Augmented lagrangian of the objective function
        Keyword Arguments:
        x     -- argument of minimisation
        lam   -- current vector of laplace multipliers
        c_pen -- current penalty
        """
        constraints_eval = constraints(x)
        return fun(x) + lam*constraints_eval + c_pen/2*(constraints_eval**2).sum()
    def mod_objective_with_args(x, lam, c_pen, *args):
        """ Augmented lagrangian of the objective function
        Keyword Arguments:
        x     -- argument of minimisation
        lam   -- current vector of laplace multipliers
        c_pen -- current penalty
        *args -- additional arguments passed to the objective function and its
                 derivatives
        """
        constraints_eval = constraints(x, *args)
        return fun(x, *args) + lam*constraints_eval + c_pen/2*(constraints_eval**2).sum()

    if args:
        mod_objective = mod_objective_with_args
    else:
        mod_objective = mod_objective_no_args

    norm_grad = 2 * tol + 1
    norm_constraint = 2 * tol + 1

    ## LANCELOT algo
    while norm_grad > tol or norm_constraint > tol:
        ## evaluate the dual objective function)
        iterate = scipy.optimize.minimize(mod_objective, x,
                                          args=(multiplier, penalty, *args),
                                          method=min_method, tol=current_tol,
                                          options=options)
        if not iterate.success:
            raise Exception(
                ("evaluation of dual objective function failed with the "
                 "following message: '{}'. The full result is\n{}").format(
                     iterate.message, iterate))
        x = iterate.x
        constraints_eval = constraints(x, *args)
        norm_constraint = np.sqrt((constraints_eval**2).sum())
        norm_grad = np.sqrt((iterate.jac**2).sum())

        ## decide whether to update the multipliers or the penalty
        if norm_grad <= update_tol:  # update multipliers
            multiplier += penalty * constraints_eval
            current_tol /= penalty
            update_tol /= penalty**beta
        else:
            penalty *= tau
            current_tol = tol/penalty
            update_tol = update_tol0/penalty**alpha
    return iterate
