#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   04-SystemTest.py

@author Till Junge <till.junge@kit.edu>

@date   11 Feb 2015

@brief  Tests the creation of tribosystems

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


You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

try:
    import unittest
    from numpy.random import rand, random
    import numpy as np

    from scipy.optimize import minimize
    from scipy.fftpack import fftn, ifftn
    import time

    import os
    from netCDF4 import Dataset

    from PyPyContact.System import SystemFactory, IncompatibleFormulationError
    from PyPyContact.System import IncompatibleResolutionError
    from PyPyContact.System.Systems import SmoothContactSystem
    import PyPyContact.SolidMechanics as Solid
    import PyPyContact.ContactMechanics as Contact
    import PyPyContact.Surface as Surface
    import PyPyContact.Tools as Tools
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class SystemTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        self.radius = 100
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

        self.substrate = Solid.PeriodicFFTElasticHalfSpace(
            self.res, self.young, self.size)

        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.smooth = Contact.LJ93smooth(self.eps, self.sig, self.gam)

        self.sphere = Surface.Sphere(self.radius, self.res, self.size)

    def test_RejectInconsistentInputTypes(self):
        with self.assertRaises(IncompatibleFormulationError):
            SystemFactory(12, 13, 24)

    def test_RejectInconsistentSizes(self):
        incompat_res = tuple((2*r for r in self.res))
        incompat_sphere = Surface.Sphere(self.radius, incompat_res, self.size)
        with self.assertRaises(IncompatibleResolutionError):
            SystemFactory(self.substrate, self.smooth, incompat_sphere)

    def test_SmoothContact(self):
        S = SmoothContactSystem(self.substrate, self.smooth, self.sphere)
        offset = self.sig
        disp = np.zeros(self.res)
        pot, forces = S.evaluate(disp, offset, forces = True)

    def test_SystemGradient(self):
        res = self.res##[0]
        size = self.size##[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(
            res, 25*self.young, self.size[0])
        sphere = Surface.Sphere(self.radius, res, size)
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        disp = random(res)*self.sig/10
        disp -= disp.mean()
        offset = self.sig
        gap = S.compute_gap(disp, offset)

        ## check subgradient of potential
        V, dV, ddV = S.interaction.evaluate(gap, pot=True, forces=True)
        f = V.sum()
        g = -dV
        fun = lambda x: S.interaction.evaluate(x)[0].sum()
        approx_g = Tools.evaluate_gradient(
            fun, gap, self.sig/1e5)

        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = ["interaction: "]
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))
        interaction = dict({"e":f,
                            "g":g,
                            "a":approx_g})
        ## check subgradient of substrate
        V, dV = S.substrate.evaluate(disp, pot=True, forces=True)
        f = V.sum()
        g = -dV
        fun = lambda x: S.substrate.evaluate(x)[0].sum()
        approx_g = Tools.evaluate_gradient(
            fun, disp, self.sig/1e5)

        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = ["substrate: "]
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))
        substrate = dict({"e":f,
                          "g":g,
                          "a":approx_g})

        V, dV = S.evaluate(disp, offset, forces=True)
        f = V
        g = -dV
        approx_g = Tools.evaluate_gradient(S.objective(offset), disp, 1e-5)
        approx_g2 = Tools.evaluate_gradient(
            lambda x: S.objective(offset, gradient=True)(x)[0], disp, 1e-5)
        tol = 1e-6
        self.assertTrue(
            Tools.mean_err(approx_g2, approx_g) < tol,
            "approx_g  = {}\napprox_g2 = {}\nerror = {}, tol = {}".format(
                approx_g, approx_g2, Tools.mean_err(approx_g2, approx_g),
                tol))


        i, s = interaction, substrate
        f_combo = i['e'] + s['e']
        error = abs(f_combo-V)
        self.assertTrue(
            error < tol,
            "f_combo = {}, f = {}, error = {}, tol = {}".format(
                f_combo, V, error, tol))


        g_combo = -i['g'] + s['g'] ## -minus sign comes from derivative of gap
        error = Tools.mean_err(g_combo, g)
        self.assertTrue(
            error < tol,
            "g_combo = {}, g = {}, error = {}, tol = {}".format(
                g_combo, g, error, tol))

        approx_g_combo = -i['a'] + s['a'] ## minus sign comes from derivative of gap
        error = Tools.mean_err(approx_g_combo, approx_g)
        self.assertTrue(
            error < tol,
            "approx_g_combo = {}, approx_g = {}, error = {}, tol = {}".format(
                approx_g_combo, approx_g, error, tol))

        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))


    def test_unconfirmed_minimization(self):
        ## this merely makes sure that the code doesn't throw exceptions
        ## the plausibility of the result is not verified
        res = self.res[0]
        size = self.size[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(
            res, 25*self.young, self.size[0])
        sphere = Surface.Sphere(self.radius, res, size)
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        offset = self.sig
        disp = np.zeros(res)

        fun_jac = S.objective(offset, gradient=True)
        fun     = S.objective(offset, gradient=False)

        info =[]
        start = time.perf_counter()
        result_grad = minimize(fun_jac, disp.reshape(-1), jac=True)
        duration_g = time.perf_counter()-start
        info.append("using gradient:")
        info.append("solved in {} seconds using {} fevals and {} jevals".format(
            duration_g, result_grad.nfev, result_grad.njev))

        start = time.perf_counter()
        result_simple = minimize(fun, disp)
        duration_w = time.perf_counter()-start
        info.append("without gradient:")
        info.append("solved in {} seconds using {} fevals".format(
            duration_w, result_simple.nfev))

        info.append("speedup (timewise) was {}".format(duration_w/duration_g))

        print('\n'.join(info))


        message = ("Success with gradient: {0.success}, message was '{0.message"
                   "}',\nSuccess without: {1.success}, message was '{1.message}"
                   "'").format(result_grad, result_simple)
        self.assertTrue(result_grad.success and result_simple.success,
                        message)

class FreeElasticHalfSpaceSystemTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        self.radius = 100
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

        self.substrate = Solid.FreeFFTElasticHalfSpace(
            self.res, self.young, self.size)

        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.smooth = Contact.LJ93smooth(self.eps, self.sig, self.gam)

        self.sphere = Surface.Sphere(self.radius, self.res, self.size)

    def test_unconfirmed_minimization(self):
        ## this merely makes sure that the code doesn't throw exceptions
        ## the plausibility of the result is not verified
        res = self.res[0]
        size = self.size[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(
            res, 25*self.young, self.size[0])
        sphere = Surface.Sphere(self.radius, res, size)
        # here, i deliberately avoid using the SystemFactory, because I want to
        # explicitly test the dumb (yet safer) way of computing problems with a
        # free, non-periodic  boundary. A user who invokes a system constructor
        # directliy like this is almost certainly mistaken
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        offset = self.sig
        disp = np.zeros(substrate.computational_resolution)

        fun_jac = S.objective(offset, gradient=True)
        fun     = S.objective(offset, gradient=False)

        info =[]
        start = time.perf_counter()
        result_grad = minimize(fun_jac, disp.reshape(-1), jac=True)
        duration_g = time.perf_counter()-start
        info.append("using gradient:")
        info.append("solved in {} seconds using {} fevals and {} jevals".format(
            duration_g, result_grad.nfev, result_grad.njev))

        start = time.perf_counter()
        result_simple = minimize(fun, disp)
        duration_w = time.perf_counter()-start
        info.append("without gradient:")
        info.append("solved in {} seconds using {} fevals".format(
            duration_w, result_simple.nfev))

        info.append("speedup (timewise) was {}".format(duration_w/duration_g))

        print('\n'.join(info))


        message = ("Success with gradient: {0.success}, message was '{0.message"
                   "}',\nSuccess without: {1.success}, message was '{1.message}"
                   "'").format(result_grad, result_simple)
        self.assertTrue(result_grad.success and result_simple.success,
                        message)

    def test_comparison_pycontact(self):
        tol = 1e-9
        ref_fpath = 'tests/ref_smooth_sphere.nc'
        out_fpath = 'tests/ref_smooth_sphere.out'
        ref_data =  Dataset(ref_fpath, mode='r', format='NETCDF4')
        with open(out_fpath) as fh:
            fh.__next__()
            fh.__next__()
            ref_N = float(fh.__next__().split()[0])
        ## 1. replicate potential
        sig = ref_data.lj_sigma
        eps = ref_data.lj_epsilon
        # pycontact doesn't store gamma (work of adhesion) in the nc file, but
        # the computed rc1, which I can test for consistency
        gamma = 0.001
        potential = Contact.LJ93smooth(eps, sig, gamma)
        error = abs(potential.r_c- ref_data.lj_rc2)
        self.assertTrue(
            error < tol,
            ("computation of lj93smooth cut-off radius differs from pycontact "
             "reference: PyPyContact: {}, pycontact: {}, error: {}, tol: "
            "{}").format(potential.r_c, ref_data.lj_rc2, error, tol))
        ## 2. replicate substrate
        res = (ref_data.size//2, ref_data.size//2)

        size= tuple((float(r) for r in res))
        young = 2. # pycontact convention (hardcoded)
        substrate = Solid.FreeFFTElasticHalfSpace(res, young, size)

        ## 3. replicate surface
        radius = ref_data.Hertz
        centre = (15.5, 15.5)
        surface = Surface.Sphere(radius, res, size, centre=centre)
        ## ref_h = -np.array(ref_data.variables['h'])
        ## ref_h -= ref_h.max()
        ## surface = Surface.NumpySurface(ref_h)
        ## 4. Set up system:
        S = SmoothContactSystem(substrate, potential, surface)

        ref_profile = np.array(
            ref_data.variables['h']+ref_data.variables['avgh'][0])[:32, :32]
        offset = .8*potential.r_c
        gap = S.compute_gap(np.zeros(substrate.computational_resolution), offset)
        diff = ref_profile-gap
        # pycontact centres spheres at (n + 0.5, m + 0.5). need to correct for test
        correction = np.sqrt(radius**2-.5)-radius
        error = Tools.mean_err(ref_profile-correction, gap)
        self.assertTrue(
            error < tol,
            "initial profiles differ (mean error ē = {} > tol = {})".format(
                error, tol))
        # pycontact does not save the offset in the nc, so this one has to be
        # taken on faith
        fun = S.objective(offset+correction, gradient=True)
        fun_hard = S.objective(offset, gradient=False)

        ## initial guess (cheating) is the solution of pycontact
        disp = np.zeros(S.substrate.computational_resolution)
        disp[:ref_data.size, :ref_data.size] = ref_data.variables['u'][0]

        options = dict(ftol = 1e-12, gtol = 1e-10)
        result = minimize(fun, disp, jac=True, callback=S.callback(force=True), method = 'L-BFGS-B', options=options)

        e, force = fun(result.x)



        error = abs(ref_N - S.compute_normal_force())
        # here the answers differ slightly, relaxing the tol for this one
        ftol = 1e-7
        self.assertTrue(
            error < ftol,
            "resulting normal forces differ: error = {} > tol = {}".format(
                error, ftol))
        error = Tools.mean_err(
            disp, result.x.reshape(S.substrate.computational_resolution))
        self.assertTrue(
            error < tol,
            "resulting displacements differ: error = {} > tol = {}".format(
                error, tol))

    def test_size_insensitivity(self):
        tol = 1e-6
        ref_fpath = 'tests/ref_smooth_sphere.nc'
        out_fpath = 'tests/ref_smooth_sphere.out'
        ref_data =  Dataset(ref_fpath, mode='r', format='NETCDF4')
        with open(out_fpath) as fh:
            fh.__next__()
            fh.__next__()
            ref_N = float(fh.__next__().split()[0])
        ## 1. replicate potential
        sig = ref_data.lj_sigma
        eps = ref_data.lj_epsilon
        # pycontact doesn't store gamma (work of adhesion) in the nc file, but
        # the computed rc1, which I can test for consistency
        gamma = 0.001
        potential = Contact.LJ93smooth(eps, sig, gamma)
        error = abs(potential.r_c- ref_data.lj_rc2)
        self.assertTrue(
            error < tol,
            ("computation of lj93smooth cut-off radius differs from pycontact "
             "reference: PyPyContact: {}, pycontact: {}, error: {}, tol: "
            "{}").format(potential.r_c, ref_data.lj_rc2, error, tol))
        nb_compars = 3
        normalforce = np.zeros(nb_compars)
        options = dict(ftol = 1e-12, gtol = 1e-10)

        for i, resolution in ((i, ref_data.size//4*2**i) for i in range(nb_compars)):
            res = (resolution, resolution)

            size= tuple((float(r) for r in res))
            young = 2. # pycontact convention (hardcoded)
            substrate = Solid.FreeFFTElasticHalfSpace(res, young, size)

            ## 3. replicate surface
            radius = ref_data.Hertz
            surface = Surface.Sphere(radius, res, size)

            ## 4. Set up system:
            S = SmoothContactSystem(substrate, potential, surface)
            # pycontact does not save the offset in the nc, so this one has to be
            # taken on faith
            offset = .8*potential.r_c
            fun = S.objective(offset, gradient=True)
            disp = np.zeros(np.prod(res)*4)
            result = minimize(fun, disp, jac=True,
                              method = 'L-BFGS-B', options=options)
            normalforce[i] = S.interaction.force.sum()
            error = abs(normalforce-normalforce.mean()).mean()
        self.assertTrue(error < tol, "error = {:.15g} > tol = {}, N = {}".format(
            error, tol, normalforce))
