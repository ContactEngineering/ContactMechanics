#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   AugmentedLagrangian.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Implements the AugmentedLagrangian as described in
        Bierlaire (2006):
        Introduction à l'optimization différentiable, Michel Bierlaire, Presses
        polytechniques et universitaires romandes, Lausanne 2006,
        ISBN:2-88074-669-8

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
import scipy.optimize
from copy import deepcopy
from . import ReachedTolerance, ReachedMaxiter, FailedIterate

# implemented as a custom minimizer for scipy
def augmented_lagrangian(fun, x0, args=(), constraints=None, tol=1e-5,
                         update_tol0=.1, multiplier0=None, penalty0=10, alpha=0.1,
                         beta=0.9, tau=10, min_method='L-BFGS-B', callback=None,
                         bounds=None, jac=None, hessp=None,
                         hess=None, store_iterates=None, **options):
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
                   constraints(x, args=args) and returns an ndarray of same
                   length as multiplier0
    tol         -- (default None) Convergence tolerance. Is used for both the
                   subjacent minimizer as well as for the outer (augmented
                   Lagrangian) loop.
    update_tol0 -- (default .1) This value is used to decide whether the
                   minimum x_k is "sufficiently" admissible: if
                   ||constraints(x_k, args=args)|| < current update_tol, then
                   the multipliers are updated, else the penalty is increased.
    multiplier0 -- (default None) Initial guess for the Lagrange multipliers
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
    callback    -- (default None) Called after each iteration, as callback(xk),
                   where xk is the current parameter vector.
    bounds      -- (default None) Bounds for variables (only for L-BFGS-B, TNC
                   and SLSQP). (min, max) pairs for each element in x, defining
                   the bounds on that parameter. Use None for one of min or max
                   when there is no bound in that direction.
    jac         -- (default None) Jacobian (gradient) of objective function.
                   Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg,
                   trust-ncg. If jac is a Boolean and is True, fun is assumed
                   to return the gradient along with the objective function. If
                   False, the gradient will be estimated numerically. jac can
                   also be a callable returning the gradient of the objective.
                   In this case, it must accept the same arguments as fun.
    hess/hessp  -- (default None) Hessian (matrix of second-order derivatives)
                   of objective function or Hessian of objective function times
                   an arbitrary vector p. Only for Newton-CG, dogleg,
                   trust-ncg. Only one of hessp or hess needs to be given. If
                   hess is provided, then hessp will be ignored. If neither
                   hess nor hessp is provided, then the Hessian product will be
                   approximated using finite differences on jac. hessp must
                   compute the Hessian times an arbitrary vector.
    store_iterates -- (default None) if set to 'iterate' the full iterates are
                   stored in module-level constant iterates
    **options   -- are handed through to min_method as is.
    """
    x = x0
    mandatory_items = {'outer_maxiter': 20,
                       'disp': False}
    for key, val in mandatory_items.items():
        if not key in options.keys():
            options[key] = val

    if not isinstance(multiplier0, np.ndarray):
        raise Exception(
            "for sanity reasons, imma require multiplier0 to be an array, even "
            "if it's scalar")
    multiplier = np.array(multiplier0, dtype=float)  # 'lam' in the objective
    penalty = penalty0        # 'c_pen' in the objective
    update_tol0 = penalty**alpha*update_tol0
    update_tol = update_tol0/penalty**alpha
    current_tol = tol
    constraints = constraints['fun']

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


    # some of the option args get duplicated in _minimize.py (annoying design
    # choice in scipy)
    # del options['bounds']

    norm_grad = 2 * tol + 1
    norm_constraint = 2 * tol + 1

    # LANCELOT algo
    counter = 0
    inner_options = deepcopy(options)
    inner_options['disp']= False
    del inner_options['outer_maxiter']

    if options['disp']:
        print(("{0[k]} | {0[x]} {0[lam]} {0[c]} {0[cur_tol]} "
               "{0[update_tol]} {0[nit]}").format(
                   {'k': counter,
                    'x': x,
                    'lam': multiplier,
                    'c': penalty,
                    'cur_tol': current_tol,
                    'update_tol': update_tol,
                    'nit': '?'}))
    iterates = list()
    try:
        while True:
            if (norm_grad < tol and norm_constraint < tol):
                raise ReachedTolerance((
                    "{0} (norm_grad) < {2} (tol) and {1} (norm_constraint) < "
                    "{2} (tol)").format(norm_grad, norm_constraint, tol))
            if counter == options['outer_maxiter']:
                raise ReachedMaxiter("reached maxiter ({})".format(
                    options['outer_maxiter']))

            ## evaluate the dual objective function)
            iterate = scipy.optimize.minimize(mod_objective, x,
                                              args=(multiplier, penalty) + args,
                                              method=min_method, tol=current_tol,
                                              bounds=bounds, jac=jac,
                                              options=inner_options)
            if store_iterates == 'iterate':
                iterates.append(iterate)
            if not iterate.success:
                raise FailedIterate(
                    ("evaluation of dual objective function failed with the "
                     "following message: '{}'. The full result is\n{}").format(
                         iterate.message, iterate))
            constraints_eval = constraints(x, *args)
            norm_constraint = np.sqrt((constraints_eval**2).sum())
            norm_grad = np.sqrt((iterate.jac**2).sum())
            x = iterate.x

            ## decide whether to update the multipliers or the penalty
            if norm_constraint <= update_tol:  # update multipliers
                multiplier += penalty * constraints_eval
                current_tol /= penalty
                update_tol /= penalty**beta
            else:
                penalty *= tau
                current_tol = tol/penalty
                update_tol = update_tol0/penalty**alpha

            # run callback, usually a noop
            counter += 1
            if callback is not None:
                callback(x)

            # possibly swamp stdout
            if options['disp']:
                print(("{0[k]} | {0[x]} {0[lam]} {0[c]} {0[cur_tol]} "
                       "{0[update_tol]} {0[nit]}").format(
                           {'k': counter,
                            'x': x,
                            'lam': multiplier,
                            'c': penalty,
                            'cur_tol': current_tol,
                            'update_tol': update_tol,
                            'nit': iterate.nit}))

    except (FailedIterate, ReachedMaxiter) as err:
        message = str(err)
        success = False
    except(ReachedTolerance) as err:
        message = str(err)
        success = True

    result = scipy.optimize.OptimizeResult({'message': message,
                                            'success': success,
                                            'x': iterate.x,
                                            'fun': iterate.fun,
                                            'jac': iterate.jac,
                                            'nit': counter})
    if iterates:
        result['iterates'] = iterates

    return result
