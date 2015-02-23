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

def augmented_lagrangian(objective_fun, eq_const_fun, solver='hybr', options=None):
    """
    Keyword Arguments:
    objective_fun --
    eq_const_fun  --
    solver        -- (default 'hybr')
    options       -- (default dict())
    """
    options = dict() if options is None else options

def augmented_lagrangian(fun, x0, constraints, mu_0=None, lamb_0=None,
                         method=None, jac=False, tol=None, callback=None,
                         options=None):
    """
    Implements the augmented lagrangian for equality constraints. The arguments
    have the same meaning as in scipy.optimize.minimize

    Keyword Arguments:
    fun         -- Objective function.
    x0          -- Initial guess.
    constraints -- a callable function c for which c(x) = 0
    method      -- (default None)
    jac         -- (default False)
    tol         -- (default None)
    callback    -- (default None)
    options     -- (default None)
    """
    mu = 1. if mu_0 is None else mu_0
    lamb = np.ones_like(lamb_0) if lamb_0 is None else lamb_0

    def objective():
        " "
        pass
