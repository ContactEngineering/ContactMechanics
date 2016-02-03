#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   08-higher_minimisation_tests.py

@author Till Junge <till.junge@kit.edu>

@date   30 Mar 2015

@brief  Tests for higher system methods, such as computing pulloff-force

@section LICENCE

 Copyright (C) 2015 Till Junge

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
try:
    import unittest
    import numpy as np
    from scipy.optimize import minimize
    import time
    import math

    from PyCo.System import SystemFactory
    from PyCo.Surface import Sphere
    from PyCo.ContactMechanics import LJ93smoothMin as LJ_pot
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace as Substrate

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)


class PulloffTest(unittest.TestCase):
    def setUp(self):
        base_res = 64
        res = (base_res, base_res)
        base_size = 100.
        size = (base_size, base_size)
        young = 10

        self.substrate = Substrate(res, young, size)

        radius = size[0]/10
        self.surface = Sphere(radius, res, size, standoff=float('inf'))

        sigma = radius/10
        epsilon = sigma * young/100
        self.pot = LJ_pot(epsilon, sigma)


    def tst_FirstContactThenOffset(self):
        system = SystemFactory(self.substrate, self.pot, self.surface)
        offset0 = .5*self.pot.r_min + .5*self.pot.r_c
        disp0 = np.zeros(self.substrate.computational_resolution)

        def obj_fun(offset):
            nonlocal disp0
            inner = None
            print("Running with offset = {}".format(offset))
            def callback(xk):
                nonlocal inner
                if inner is None:
                    inner = system.callback(True)
                inner(xk)

            options = {'gtol':1e-20, 'ftol': 1e-20}
            result = system.minimize_proxy(offset, disp0, #callback=callback,
                                           options=options)
            if result.success:
                disp0 = system.disp
                return system.compute_normal_force()
            diagnostic = "offset = {}, r_c = {}, mean(f_pot) = {}".format(
                offset, self.pot.r_c, system.interaction.force.mean())
            gap = system.compute_gap(system.disp, offset)
            diagnostic += ", gap: (min, max) = ({}, {})".format(gap.min(), gap.max())
            diagnostic += ", disp: (min, max) = ({}, {})".format(system.disp.min(), system.disp.max())

            raise Exception("couldn't minimize: {}: {}".format(result.message, diagnostic))
        #constrain the force to be negative by convention
        def fun(x):
            retval = -obj_fun(x)
            print("WILL BE RETURNING: {}".format(retval))
            return retval
        constraints = ({'type': 'ineq', 'fun': fun}, )
        bounds = ((0.7, 1),)
        start = time.perf_counter()
        result = minimize(obj_fun, x0=offset0, constraints=constraints, bounds = bounds, method='slsqp', options={'ftol':1e-8})
        duration = time.perf_counter()-start
        msg = str(result)
        msg += "\ntook {} seconds".format(duration)
        raise Exception(msg)

    def tst_FirstOffsetThenContact(self):
        system = SystemFactory(self.substrate, self.pot, self.surface)
        offset0 = .5*self.pot.r_min + .5*self.pot.r_c
        disp0 = np.zeros(self.substrate.computational_resolution)

        def minimize_force(offset0, const_disp):
            system.create_babushka(offset0, const_disp)
            min_gap = system.compute_gap(const_disp, 0).min()
            # bounds insure non-penetration and non-separation
            bounds = ((-min_gap, .99*self.pot.r_c-min_gap), )
            def obj(offset):
                babushka_disp = system._get_babushka_array(const_disp)
                system.babushka.evaluate(babushka_disp, offset,
                                         pot=False, forces=True)
                return system.babushka.compute_normal_force()
            result = minimize(obj, x0=offset0, bounds=bounds)

            if result.success:
                return result.x, result.fun
            raise Exception(str(result))
        result = minimize_force(offset0, disp0)

        r_range = self.pot.r_c-self.pot.r_min
        tol = r_range*1e-8
        delta_offset = tol+1.
        max_iter = 100
        it = 0
        disp = disp0
        offset = offset0
        while delta_offset > tol and it < max_iter:
            it +=1
            new_offset, pulloff_force = minimize_force(offset, disp)
            signed_delta = offset - new_offset
            delta_offset = abs(signed_delta)
            offset -= min(delta_offset, r_range*.01)*math.copysign(1., signed_delta)
            result = system.minimize_proxy(offset, disp)
            disp = system.disp
            message = "iteration {:>3}: ".format(it)
            if result.success:
                message += "success"
            else:
                message += "FAIL!!"
            message += ", delta_offset = {}, force = {}".format(delta_offset, pulloff_force)
            print(message)

        raise Exception("Done! force = {}, offset = {} (offset_max = {}),\nresult = {}".format(
            pulloff_force, offset, self.pot.r_c, result))

