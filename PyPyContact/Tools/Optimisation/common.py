#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   common.py

@author Till Junge <till.junge@kit.edu>

@date   17 Sep 2015

@brief  Common helpers and exception definitions for optimisers

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
import warnings
import numpy as np
import scipy.optimize


class ReachedTolerance(StopIteration):
    # pylint: disable=missing-docstring
    pass


class ReachedMaxiter(StopIteration):
    # pylint: disable=missing-docstring
    pass


class FailedIterate(StopIteration):
    # pylint: disable=missing-docstring
    pass


class ReachedMaxiterWarning(RuntimeWarning):
    # pylint: disable=missing-docstring
    pass


# The following helpers are from
# Bierlaire (2006):
# Introduction à l'optimization différentiable, Michel Bierlaire, Presses
# polytechniques et universitaires romandes, Lausanne 2006,
# ISBN:2-88074-669-8

# -----------------------------------------------------------------------------
# Algorithm 12.1 (p. 297) Intersection with confidence region
def intersection_confidence_region(x_start, direction, radius):
    """
    Find the intersection between a direction and the boundary of the
    confidence region

    returns the step length

    Keyword Arguments:
    x_start   -- starting point |x| <= radius
    direction -- search direction != 0
    radius    -- scalar > 0
    """
    # pylint: disable=invalid-name
    a = float(direction.T * direction)
    b = float(2 * x_start.T * direction)
    c = float(x_start.T * x_start - radius**2)
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)


# -----------------------------------------------------------------------------
# Algorithm 12.2 (p.297) dogleg method
def dogleg(grad_f, hess_f, radius):
    """
    Finds an approximation to the solution of the confidence region sub-problem
    Keyword Arguments:
    grad_f -- current value of the gradient (column vector)
    hess_f -- value of the hessian matrix (square matrix)
    radius -- radius of the confidence region
    """
    # Cauchy point
    # 1) Computation of the curvature in steepest descent direction
    beta = float(grad_f.T * hess_f * grad_f)

    # 2) If beta <=0, the model is not locally convex.
    if beta <= 0:
        return -radius / np.linalg.norm(grad_f) * grad_f

    # 3) else, compute the Cauchy point
    alpha = float(grad_f.T * grad_f)
    d_c = - alpha / beta * grad_f

    # 4) make sure we're still in the confidence interval
    step_len_c = np.linalg.norm(d_c)
    if step_len_c > radius:
        return radius / step_len_c * d_c

    # Newton point
    # 1) compute directly
    d_n = np.linalg.solve(hess_f, -grad_f)

    # 2) If not convex, stop with Cauchy point
    if float(d_n.T * hess_f * d_n) <= 0:
        return d_c

    # 3) if d_n in region, return with it
    if np.linalg.norm(d_n) <= radius:
        return d_n

    # Dogleg point
    # 1)
    eta = 0.2 + (0.8 * alpha**2) / (beta * float(abs(grad_f.T * d_n)))
    d_d = eta * d_n

    # 2)
    if np.linalg.norm(d_d) <= radius:
        return radius / np.linalg.norm(d_n) * d_n

    # between Cauchy and dogleg
    # 1) compute the intersection
    step_len = intersection_confidence_region(d_c, d_d - d_c, radius)

    return d_c + step_len * (d_d - d_c)


# -----------------------------------------------------------------------------
# Algorithm 12.3 (p.297) Steihaug-Toint method
def steihaug_toint(grad_f, hess_f, radius, tol=1e-14):
    """
    Finds an approximation to the solution of the confidence region sub-problem
    Keyword Arguments:
    grad_f -- current value of the gradient (column vector)
    hess_f -- value of the hessian matrix (square matrix)
    radius -- radius of the confidence region
    """
    # pylint: disable=invalid-name
    # initialisation
    direction = -grad_f.copy()
    xk = np.zeros_like(direction)
    xkp1 = np.zeros_like(direction)

    for _ in range(grad_f.size + 1):
        # 1) Computation of the curvature in steepest descent direction
        if float(direction.T * hess_f * direction) <= 0:
            step_len = intersection_confidence_region(xk, direction, radius)
            return xk + step_len * direction

        # 2) Compute step length
        alpha = (- float(direction.T*(hess_f * xk + grad_f)) /
                 float(direction.T * hess_f * direction))

        # 3) compute next iterate
        xkp1 = xk + alpha * direction

        # 4)
        if np.linalg.norm(xkp1) > radius:
            step_len = intersection_confidence_region(xk,
                                                      direction, radius)
            return xk + step_len * direction

        # 5) compute beta
        grad_k = hess_f * xk + grad_f
        grad_kp1 = hess_f * xkp1 + grad_f
        beta = float((grad_kp1.T * grad_kp1) / (grad_k.T * grad_k))

        # 6) compute new direction
        direction = - grad_kp1 + beta * direction

        if np.linalg.norm(grad_kp1) < tol:
            return xkp1

        # 7) cleanup
        xk[:] = xkp1[:]
    return xk


# -----------------------------------------------------------------------------
# implements the modified Cholesky Factorisation, p. 278, algo 11.4
def modified_cholesky(symmat, maxiter=20):
    """
    Modify a symmetric matrix A in order to make it positive definite. Returns
    a lower triangular matrix L and a scalar τ > 0 so that
              A + τI = LL^T
    Keyword Arguments:
    symmat -- symmetric matrix
    """
    fronorm = np.linalg.norm(symmat, ord='fro')
    if np.diag(symmat).min() > 0:
        tau = 0
    else:
        tau = .5*fronorm
    I = np.eye(symmat.shape[0])  # pylint: disable=invalid-name

    for _ in range(maxiter):
        try:
            L = np.linalg.cholesky(  # pylint: disable=invalid-name
                symmat + tau*I)
            return L, tau
        except np.linalg.LinAlgError:
            tau = max(2*tau, .5*fronorm)
    raise Exception("Couldn't factor")


# -----------------------------------------------------------------------------
def first_wolfe_condition(fun, x0, fprime, direction, alpha, beta1):
    """
    p. 268, 11.19

    Keyword Arguments:
    fun         -- objective function to minimize
    x0          -- initial guess for solution
    fprime      -- Jacobian (gradient)
    direction   -- search direction (column vec)
    alpha       -- step size
    beta1       -- lower wolfe bound
    """
    return float(fun(x0+alpha*direction)) <= float(fun(x0)) + \
        alpha * beta1 * float(fprime(x0).T * direction)


# -----------------------------------------------------------------------------
def second_wolfe_condition(x0, fprime, direction, alpha, beta2):
    """
    p. 270, 11.21

    Keyword Arguments:
    x0        -- initial guess for solution
    fprime    -- Jacobian (gradient) of objective function
    direction -- search direction
    alpha     -- step size
    beta2     -- upper wolfe bound
    """
    return (float(fprime(x0 + alpha*direction).T * direction) >=
            beta2*float(fprime(x0).T * direction))


# -----------------------------------------------------------------------------
# implements the line search, p. 273, algo 11.2
def line_search(fun, x0, fprime, direction, alpha0, beta1=1e-4, beta2=0.99,
                step_factor=3., store_iterates=None, maxiter=40):
    """
    find a step size alpha that satisfies both conditions of Wolfe
    Keyword Arguments:
    fun         -- objective function to minimize
    x0          -- initial guess for solution
    fprime      -- Jacobian (gradient) of objective function
    direction   -- search direction
    alpha0      -- Initial guess for step size
    beta1       -- (default 1e-4)
    beta2       -- (default 0.99)
    step_factor -- (default 3.) step increase when too short
    store_iterates -- (default None) if set to 'iterate' the full iterates are
                   stored in module-level constant iterates
    maxiter     -- (default 20) abort and raise Exception after maxiter is
                   reached
    """
    alpha_l = 0
    alpha_r = float('inf')
    alpha = alpha0

    wolfe1 = first_wolfe_condition(fun, x0, fprime, direction, alpha, beta1)
    wolfe2 = second_wolfe_condition(x0, fprime, direction, alpha, beta2)

    iterates = list()
    counter = 0
    violation = 0
    if store_iterates == 'iterate':
        iterate = scipy.optimize.OptimizeResult(
            {'x': x0.copy(),
             'fun': fun(x0),
             'jac': fprime(x0),
             'alpha_i': alpha,
             'alpha_r': alpha_r,
             'alpha_l': alpha_l,
             'violation': 0})
        iterates.append(iterate)
    while not (wolfe1 and wolfe2):
        if counter == maxiter:
            warnings.warn(
                ("Line search did not converge. Are your jacobians correct? "
                 "wolfe1 = {}, wolfe2 = {}, alpha = {}, nit = {}.\n"
                 "If they are, machine precision has been reached. Currently,"
                 " progress regarding funval would be {}").format(
                     wolfe1, wolfe2, alpha, counter,
                     float(alpha * fprime(x0).T*direction)),
                ReachedMaxiterWarning)
            break
        if not wolfe1:  # step too long
            alpha_r = alpha
            alpha = .5*(alpha_l + alpha_r)
            violation = 1

        elif wolfe1 and not wolfe2:
            alpha_l = alpha
            violation = 2
            if np.isfinite(alpha_r):
                alpha = .5*(alpha_l + alpha_r)
            else:
                alpha *= step_factor
        wolfe1 = first_wolfe_condition(fun, x0, fprime, direction, alpha,
                                       beta1)
        wolfe2 = second_wolfe_condition(x0, fprime, direction, alpha, beta2)
        if store_iterates == 'iterate':
            iterate = scipy.optimize.OptimizeResult(
                {'x': x0.copy(),
                 'fun': fun(x0),
                 'jac': fprime(x0),
                 'alpha_i': alpha,
                 'alpha_r': alpha_r,
                 'alpha_l': alpha_l,
                 'violation': violation})
            iterates.append(iterate)
        counter += 1

    result = scipy.optimize.OptimizeResult({'success': True,
                                            'x': alpha,
                                            'nit': counter,
                                            'violation': violation})

    if iterates:
        result['iterates'] = iterates
    return result


# -----------------------------------------------------------------------------
# Langrangian definitions
def construct_augmented_lagrangian(fun, constraints):
    """
    According to 6.17, p. 164:
    Given a function f(x) and equality constraints h(x)=0 this constructs and
    returns the augmented Lagrangian L(x, λ, c).
    Keyword Arguments:
    fun           -- objective function to minimize
    constraints   -- vector of equality constraint functions. A callable
                     that is called as constraints(x, args=args) and returns
                     an ndarray of same length as multiplier0. Rⁿ→Rᵐ
    """
    def objective(x, lam, c_pen, *args):
        """ Augmented lagrangian of the objective function
        Keyword Arguments:
        x     -- argument of minimisation
        lam   -- current vector of laplace multipliers
        c_pen -- current penalty
        *args -- additional arguments passed to the objective function and its
                 derivatives
        """
        x = np.matrix(x, copy=False).reshape((-1, 1))
        constraints_eval = constraints(x, *args)
        return (fun(x, *args) + float(
            (lam.T*constraints_eval +
             c_pen/2*(constraints_eval.T*constraints_eval))))
    return objective


def construct_augm_lag_grad(fun_jac, constraints_jac, constraints):
    """
    According to 20.4, p. 448:
    Keyword Arguments:
    fun_jac         -- gradient of objective function to minimize
    constraints_jac -- gradient of constraints Rⁿ→Rᵐˣⁿ
    constraints     -- vector of equality constraint functions. A callable
                       that is called as constraints(x, args=args) and returns
                       an ndarray of same length as multiplier0. Rⁿ→Rᵐ
    """
    def jac(x, lam, c_pen, *args):
        """
        jacobian of the augmented lagrangian of the objective function
        Keyword Arguments:
        x     -- argument of minimisation
        lam   -- current vector of laplace multipliers
        c_pen -- current penalty
        *args -- additional arguments passed to the objective function and its
        derivatives
        """
        x = np.matrix(x, copy=False).reshape((-1, 1))
        constraints_eval = constraints(x, *args)
        constraints_jac_eval = constraints_jac(x, *args)
        retjac = fun_jac(x, *args) + constraints_jac_eval.T*lam + c_pen * \
            constraints_jac_eval.T*constraints_eval
        if retjac.dtype.kind not in np.typecodes['AllFloat']:
            raise Exception("jac =\n{}\nlam_term =\n{}\nc_term =\n{}".format(
                jac(x, *args), constraints_jac_eval.T*lam,
                c_pen * constraints_jac_eval.T*constraints_eval))
        return retjac
    return jac


def construct_augm_lag_hess(fun_hess, constraints_hess,
                            constraints_jac, constraints):
    """
    According to 20.5, p. 448:
    Keyword Arguments:
    fun_hess         -- hessian of objective function to minimize
    constraints_jac  -- gradient of constraints Rⁿ→Rᵐˣⁿ
    constraints_hess -- hessian of constraints Rⁿ→Rᵐˣⁿˣⁿ
    constraints      -- vector of equality constraint functions. A callable
                        that is called as constraints(x, args=args) and returns
                        an ndarray of same length as multiplier0. Rⁿ→Rᵐ
    """
    def lag_hess(x, lam, c_pen, *args):
        """
        jacobian of the augmented lagrangian of the objective function
        Keyword Arguments:
        x     -- argument of minimisation
        lam   -- current vector of laplace multipliers
        c_pen -- current penalty
        *args -- additional arguments passed to the objective function and its
                 derivatives
        """
        x = np.matrix(x, copy=False).reshape((-1, 1))
        constraints_eval = constraints(x, *args)
        dummy = constraints_eval
        constraints_jac_eval = constraints_jac(x, *args)
        constraints_hess_eval = constraints_hess(x, *args)
        return_hess = fun_hess(x, *args) + c_pen * constraints_jac_eval.T * \
            constraints_jac_eval
        for i in range(lam.size):
            return_hess += (lam[i] * constraints_hess_eval[i] +
                            c_pen*constraints_jac_eval *
                            constraints_jac_eval.T)
            return return_hess
    return lag_hess
