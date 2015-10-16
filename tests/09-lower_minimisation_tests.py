#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   09-lower_minimisation_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   17 Sep 2015

@brief  Tests the basic minimisation functions

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

try:
    import unittest
    import numpy as np
    from scipy.optimize import minimize

    from PyPyContact.Tools.Optimisation import intersection_confidence_region
    from PyPyContact.Tools.Optimisation import dogleg
    from PyPyContact.Tools.Optimisation import steihaug_toint
    from PyPyContact.Tools.Optimisation import modified_cholesky
    from PyPyContact.Tools.Optimisation import first_wolfe_condition
    from PyPyContact.Tools.Optimisation import second_wolfe_condition
    from PyPyContact.Tools.Optimisation import line_search
    from PyPyContact.Tools.Optimisation import augmented_lagrangian
    from PyPyContact.Tools.Optimisation import newton_linesearch
    from PyPyContact.Tools.Optimisation import newton_confidence_region
    from PyPyContact.Tools import mean_err

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


# -----------------------------------------------------------------------------
class NewtonConfidenceRegionTest(unittest.TestCase):

    def setUp(self):
        def fun(x):
            return .5*x[0]**2 + 4.5 * x[1]**2
        def grad_f(x):
            return np.matrix([x[0, 0], 9*x[1, 0]]).T
        def hess_f(x):
            return np.matrix([[1, 0], [0, 9]])

        self.fun = fun
        self.grad_f = grad_f
        self.hess_f = hess_f


    def test_intersection_confidence_region(self):
        x_start = np.matrix([0., 0]).T
        direction = np.matrix([0, -2.]).T
        radius = 4

        step_len = intersection_confidence_region(x_start, direction, radius)
        self.assertTrue(step_len == 2)

    def test_dogleg(self):
        x0 = np.matrix([9, 1]).T
        f = self.fun(x0)
        grad = self.grad_f(x0)
        hess = self.hess_f(x0)

        nb_steps = 16
        radii = np.linspace(0, 10, nb_steps+1)[1:]
        iterates = np.zeros((nb_steps, 2))
        for i, radius in enumerate(radii):
            iterates[i, :] = (dogleg(grad, hess, radius) + x0).T
        solution = np.array([[ 8.55805826,  0.55805826],
                             [ 8.11611652,  0.11611652],
                             [ 7.67417479, -0.32582521],
                             [ 7.23223305, -0.76776695],
                             [ 6.14232041, -0.26463132],
                             [ 5.35089824,  0.13596509],
                             [ 4.65504028,  0.48818949],
                             [ 4.03058133,  0.44784237],
                             [ 3.40940399,  0.37882267],
                             [ 2.78822666,  0.30980296],
                             [ 2.16704932,  0.24078326],
                             [ 1.54587199,  0.17176355],
                             [ 0.92469466,  0.10274385],
                             [ 0.30351732,  0.03372415],
                             [ 0.        ,  0.        ],
                             [ 0.        ,  0.        ]])

        tol = 1e-7
        error = mean_err(iterates, solution)
        self.assertTrue(error < tol)

    def test_steihaug_toint(self):
        x0 = np.matrix([9., 1.]).T
        f = self.fun(x0)
        grad = self.grad_f(x0)
        hess = self.hess_f(x0)

        nb_steps = 5
        radii = np.linspace(0, 10, nb_steps+1)[1:]
        iterates = np.zeros((nb_steps, 2))
        for i, radius in enumerate(radii):
            iterates[i, :] = (steihaug_toint(grad, hess, radius) + x0).T

        solution = np.array([[  7.58578644e+00, -4.14213562e-01],
                             [  5.33058286e+00, -5.92286984e-01],
                             [  3.15394875e+00, -3.50438750e-01],
                             [  1.07876863e+00, -1.19863181e-01],
                             [  0.00000000e+00,  2.22044605e-16]])
        tol = 1e-7
        error = mean_err(iterates, solution)
        self.assertTrue(error < tol)

    def test_newton_confidence_dogleg(self):
        def fun(x):
            return .5*x[0, 0]**2 + x[0, 0]* np.cos(x[1, 0])

        def jac(x):
            return np.array([[x[0, 0] + np.cos(x[1, 0])],
                             [-x[0, 0] * np.sin(x[1, 0])]])

        def hess(x):
            return np.array([[           1.,        -np.sin(x[1, 0])],
                             [-np.sin(x[1, 0]), -x[0, 0] * np.cos(x[1, 0])]])
        x0 = np.array([1.,1.])
        result = minimize(fun, x0=x0, method=newton_confidence_region,
                          jac=jac, hess=hess, tol=1.e-8,
                          options={'store_iterates': 'iterate',
                                   'method': dogleg,
                                   'radius0': 1})

        solution = np.array([[  1.00000000e+00,  1.00000000e+00],
                             [  1.22417438e-01,  1.47942554e+00],
                             [ -1.01628508e-03,  1.57003091e+00],
                             [ -5.36408473e-04,  1.56808690e+00],
                             [ -5.08985118e-03,  1.56696289e+00],
                             [ -2.68657257e-03,  1.55722709e+00],
                             [ -2.54882093e-02,  1.55159842e+00],
                             [ -1.34638337e-02,  1.50289394e+00],
                             [ -1.27229935e-01,  1.47479503e+00],
                             [ -6.84750086e-02,  1.23764063e+00],
                             [ -5.88466219e-01,  1.10749811e+00],
                             [ -4.02532665e-01,  4.16075205e-01],
                             [ -4.02532665e-01,  4.16075205e-01],
                             [ -1.09349862e+00, -4.33823827e-01],
                             [ -1.10395413e+00,  3.38628972e-02],
                             [ -1.00046611e+00,  3.16267561e-03],
                             [ -1.00000500e+00,  1.44712180e-06],
                             [ -1.00000000e+00,  7.23075033e-12]])
        tol = 1e-7
        iterates = np.array([it.x.reshape(-1) for it in result.iterates]).reshape((-1, 2))
        error = mean_err(iterates, solution)
        self.assertTrue(error < tol)

    def test_newton_confidence_steihaug_toint(self):
        def fun(x):
            return .5*x[0, 0]**2 + x[0, 0]* np.cos(x[1, 0])

        def jac(x):
            return np.array([[x[0, 0] + np.cos(x[1, 0])],
                             [-x[0, 0] * np.sin(x[1, 0])]])

        def hess(x):
            return np.array([[           1.,        -np.sin(x[1, 0])],
                             [-np.sin(x[1, 0]), -x[0, 0] * np.cos(x[1, 0])]])
        x0 = np.array([1.,1.])
        result = minimize(fun, x0=x0, method=newton_confidence_region,
                          jac=jac, hess=hess, tol=1.e-8,
                          options={'store_iterates': 'iterate',
                                   'method': steihaug_toint,
                                   'radius0': 10})

        solution = np.array([[ 1.        , 1.        ],
                             [ 1.        , 1.        ],
                             [ 1.        , 1.        ],
                             [ 0.55023001, 3.4592086 ],
                             [ 1.16790323, 2.76142209],
                             [ 1.0636517 , 3.12536183],
                             [ 1.000116  , 3.14062447],
                             [ 1.00000047, 3.14159254],
                             [ 1.        , 3.14159265]])
        tol = 1e-7
        iterates = np.array([it.x.reshape(-1) for it in result.iterates]).reshape((-1, 2))
        error = mean_err(iterates, solution)
        self.assertTrue(error < tol)


# -----------------------------------------------------------------------------
class LinesearchTest(unittest.TestCase):

    def setUp(self):
        def test_fun(x):
            x.shape=(-1, 1)
            return .5*x[0, 0]**2 + x[0, 0]* np.cos(x[1, 0])

        def test_jac(x):
            x.shape=(-1, 1)
            return np.matrix([[x[0, 0] + np.cos(x[1, 0])],
                              [-x[0, 0] * np.sin(x[1, 0])]])

        def test_hess(x):
            x.shape=(-1, 1)
            return np.matrix([[           1.,        -np.sin(x[1, 0])],
                     [-np.sin(x[1, 0]), -x[0, 0] * np.cos(x[1, 0])]])

        self.fun = test_fun
        self.grad_f = test_jac
        self.hess_f = test_hess


    def test_line_search(self):
        Q = np.matrix([[1., 0.],
                       [0., 9.]])
        obj = lambda x: float(0.5*x.T*Q*x)
        grad = lambda x: Q*x
        hess_st = lambda x: Q

        x = np.matrix([10, 1.]).T
        d = np.matrix([-2., 1,]).T/np.sqrt(5)
        alpha0 = 1e-3
        beta1, beta2 = 0.3, 0.7
        step_factor = 20
        result = line_search(obj, x, grad, d, alpha0, beta1, beta2, step_factor, store_iterates='iterate')

        reference_solution = np.array([[+1.0000e-03, +0.0000e+00],
                                       [+2.0000e-02, +1.0000e-03],
                                       [+4.0000e-01, +2.0000e-02],
                                       [+8.0000e+00, +4.0000e-01],
                                       [+4.2000e+00, +4.0000e-01],
                                       [+2.3000e+00, +4.0000e-01]])
        solution = np.array([(it.alpha_i, it.alpha_l) for it in result.iterates])
        tol = 1e-7
        error = mean_err(reference_solution, solution)
        self.assertTrue(error < tol, "error = {}, tol = {}".format(error, tol))


    def test_newton_linesearch(self):
        x0 = np.array([1.,1.])
        result = minimize(self.fun, x0=x0, method=newton_linesearch,
                          jac=self.grad_f, hess=self.hess_f, tol=1.e-14,
                          options={'store_iterates': 'iterate'})

        solution = np.array([(+1.0000000e+00, +1.0000000e+00),
                             (+5.5127702e-01, +1.4196826e+00),
                             (+5.3665860e-01, +2.1020785e+00),
                             (+8.4578336e-01, +2.7823144e+00),
                             (+1.1093558e+00, +3.2749356e+00),
                             (+1.0073162e+00, +3.1531349e+00),
                             (+1.0000657e+00, +3.1416752e+00),
                             (+1.0000000e+00, +3.1415927e+00),
                             (+1.0000000e+00, +3.1415927e+00),
                             (+1.0000000e+00, +3.1415927e+00)])
        tol = 1e-7
        iterates = np.array([it.x.reshape(-1) for it in result.iterates]).reshape((-1, 2))
        error = mean_err(iterates, solution)
        self.assertTrue(error < tol, "error = {}, tol = {}".format(error, tol))


# -----------------------------------------------------------------------------
class AugmentedLagrangianTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_example(self):
        """
        Example 20.5: Minimise the fuction $f(x)$
        $$\min_{x\in\mathbb{R}^2} 2(x_1^2+x_2^2 -1)-x_1$$
        under the constraint
        $$ x_1^2 + x_2^2 = 1$$
        """
        Q = np.matrix([[4, 0], [0, 4.]])
        b = np.matrix([[-1., 0]]).T
        c = -1.
        def fun(x):
            return .5*x.T*Q*x + x.T*b + c
        def jac(x):
            return(Q*x+b)

        def hess(x):
            return Q

        cQ = .5*Q
        cb = np.matrix(((0, 0.))).T
        cc = -1.
        def constraint(x):
            return .5*x.T*cQ*x + x.T*cb + cc
        def constraint_jac(x):
            return (cQ*x + cb).T
        def constraint_hess(x):
            return Q

        tol = 1.e-2
        print(constraint_jac(np.matrix(([-1], [.1]))))

        multiplier0 = np.matrix(((0.)))
        print("multiplier0 = {}".format(multiplier0))
        maxiter=1000
        result = minimize(
            fun, x0=np.matrix(([-1], [.1])),
       	    constraints={'type':'eq','fun':constraint},
	    method=augmented_lagrangian, tol=tol,
	    options={'multiplier0': multiplier0,
                     'store_iterates': 'iterate',
                     'maxiter':maxiter,
                     'outer_maxiter':maxiter})

