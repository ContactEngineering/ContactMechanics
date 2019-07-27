#
# Copyright 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
# 
# ### MIT license
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Tests the creation of tribosystems
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

    from PyCo.System import make_system, IncompatibleFormulationError
    from PyCo.System import IncompatibleResolutionError
    from PyCo.System.Systems import SmoothContactSystem
    import PyCo.SolidMechanics as Solid
    import PyCo.ContactMechanics as Contact
    import PyCo.Topography as Topography
    import PyCo.Tools as Tools
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")

BASEDIR = os.path.dirname(os.path.realpath(__file__))

class SystemTest(unittest.TestCase):
    def setUp(self):
        self.physical_sizes = (7.5+5*rand(), 7.5+5*rand())
        self.radius = 100
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

        self.substrate = Solid.PeriodicFFTElasticHalfSpace(self.res, self.young,
                                                           self.physical_sizes)

        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.smooth = Contact.LJ93smoothMin(self.eps, self.sig, self.gam)

        self.sphere = Topography.make_sphere(self.radius, self.res,
                                             self.physical_sizes)

    def test_RejectInconsistentInputTypes(self):
        with self.assertRaises(IncompatibleFormulationError):
            make_system(12, 13, 24)

    def test_RejectInconsistentSizes(self):
        incompat_res = tuple((2*r for r in self.res))
        incompat_sphere = Topography.make_sphere(self.radius, incompat_res,
                                                 self.physical_sizes)
        with self.assertRaises(IncompatibleResolutionError):
            make_system(self.substrate, self.smooth, incompat_sphere)

    def test_SmoothContact(self):
        S = SmoothContactSystem(self.substrate, self.smooth, self.sphere)
        offset = self.sig
        disp = np.zeros(self.res)
        pot, forces = S.evaluate(disp, offset, forces = True)

    def test_SystemGradient(self):
        res = self.res##[0]
        size = [r*1.28 for r in self.res]##[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                      size)
        sphere = Topography.make_sphere(self.radius, res, size)
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        disp = random(res)*self.sig/10
        disp -= disp.mean()
        offset = -self.sig
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
        msg.append("g/approx = {}".format(g/approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))
        interaction = dict({"e":f*S.area_per_pt,
                            "g":g*S.area_per_pt,
                            "a":approx_g*S.area_per_pt})
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


        g_combo = i['g'] + s['g']
        error = Tools.mean_err(g_combo, g)
        self.assertTrue(
            error < tol,
            "g_combo = {}, g = {}, error = {}, tol = {}, g/g_combo = {}".format(
                g_combo, g, error, tol, g/g_combo))

        approx_g_combo = i['a'] + s['a']
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
        size = self.physical_sizes[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                      self.physical_sizes[0])
        sphere = Topography.make_sphere(self.radius, res, size)
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

    def test_minimize_proxy(self):
        res = self.res
        size = self.physical_sizes
        substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                      self.physical_sizes[0])
        sphere = Topography.make_sphere(self.radius, res, size)
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        offset = self.sig
        nb_scales = 5
        n_iter = np.zeros(nb_scales, dtype=int)
        n_force = np.zeros(nb_scales, dtype=float)
        for i in range(nb_scales):
            scale = 10**(i-2)
            res = S.minimize_proxy(offset, disp_scale=scale, tol = 1e-40,
                                   gradient=True, callback=True)
            print(res.message)
            n_iter[i] = res.nit
            n_force[i] = S.compute_normal_force()
        print("N_iter = ", n_iter)
        print("N_force = ", n_force)

    def test_minimize_proxy_tol(self):
        res = self.res
        size = self.physical_sizes
        substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                      self.physical_sizes[0])
        sphere = Topography.make_sphere(self.radius, res, size)
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        offset = self.sig

        res = S.minimize_proxy(offset, tol = 1e-20,
                               gradient=True, callback=True)
        print(res.message)

        rep_force = np.where(
            S.interaction.force > 0, S.interaction.force, 0
            )
        alt_rep_force = -np.where(
            S.substrate.force < 0, S.substrate.force, 0
            )

        error = Tools.mean_err(rep_force, alt_rep_force)


        ## import matplotlib.pyplot as plt
        ## fig = plt.figure()
        ## CS = plt.contourf(S.interaction.force)
        ## plt.colorbar(CS)
        ## plt.title("interaction")
        ## fig = plt.figure()
        ## CS = plt.contourf(S.substrate.force)
        ## plt.colorbar(CS)
        ## plt.title("substrate")
        ## plt.show()

        self.assertTrue(error < 1e-5, "error = {}".format(error))

        error = rep_force.sum() - S.compute_repulsive_force()
        self.assertTrue(error < 1e-5, "error = {}".format(error))

        error = (rep_force.sum() + S.compute_attractive_force() -
                 S.compute_normal_force())
        self.assertTrue(error < 1e-5, "error = {}".format(error))


def test_LBFGSB_Hertz():
    """
    goal is that this test run the hertzian contact unsing L-BFGS-B

    For some reason it is difficult to reach the gradient tolerance

    """
    nx, ny = 64,64
    sx, sy = 20., 20.
    R = 11.

    surface =Topography.make_sphere( R,(nx,ny), (sx,sy), kind="paraboloid")
    Es=50.
    substrate = Solid.FreeFFTElasticHalfSpace((nx,ny), young=Es,
                                              physical_sizes=(sx, sy),
                                              fft="serial",
                                              communicator=MPI.COMM_SELF)

    interaction = Contact.Exponential(0., 0.0001)
    system = SmoothContactSystem(substrate, interaction,surface)

    gtol=1e-7 # 1e-8 is not reachable for some reason #FIXME
    offset=1.
    res = system.minimize_proxy(offset=offset, lbounds="auto",
                                options=dict(gtol=gtol, ftol=0))

    assert res.success, res.message
    print(res.message)
    print(np.max(abs(res.jac))) # This far beyond the tolerance because
    # at the points where the constraint act the gradient is allowed to not be zero

    padding_mask = np.full(substrate.nb_subdomain_grid_pts , True)
    padding_mask[substrate.topography_subdomain_slices] = False

    print(np.max(abs(res.jac[padding_mask])))
    #ä no force in padding area
    np.testing.assert_allclose(
        system.substrate.force[padding_mask],0, rtol=0, atol=gtol)
    comp_contact_area = np.sum(
        np.where(system.compute_gap(res.x, offset) == 0, 1.,0.)) \
                        * system.area_per_pt

    comp_normal_force= np.sum(-substrate.evaluate_force(res.x))
    from PyCo.ReferenceSolutions import Hertz as Hz
    a, p0 = Hz.radius_and_pressure(Hz.normal_load(offset, R, Es), R, Es)

    np.testing.assert_allclose(comp_normal_force,
                        Hz.normal_load(offset, R, Es),
                        rtol=1e-3,
                        err_msg="computed normal force doesn't match with hertz "
                                "theory for imposed Penetration {}".format(
                            offset))

    np.testing.assert_allclose(comp_contact_area, np.pi * a ** 2,
                        rtol=1e-2,
                        err_msg="Computed area doesn't match Hertz Theory")

    if False:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.colorbar(ax.pcolormesh(- system.substrate.force), label="pressure")
        plt.show(block=True)

class FreeElasticHalfSpaceSystemTest(unittest.TestCase):
    def setUp(self):
        self.physical_sizes = (7.5+5*rand(), 7.5+5*rand())
        self.radius = 100
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

        self.substrate = Solid.FreeFFTElasticHalfSpace(
            self.res, self.young, self.physical_sizes)

        self.eps = 1+np.random.rand()
        self.sig = 3+np.random.rand()
        self.gam = (5+np.random.rand())
        self.rcut = 2.5*self.sig+np.random.rand()
        self.smooth = Contact.LJ93smooth(self.eps, self.sig, self.gam)

        self.sphere = Topography.make_sphere(self.radius, self.res, self.physical_sizes)

    def test_unconfirmed_minimization(self):
        ## this merely makes sure that the code doesn't throw exceptions
        ## the plausibility of the result is not verified
        res = self.res[0]
        size = self.physical_sizes[0]
        substrate = Solid.PeriodicFFTElasticHalfSpace(res, 25 * self.young,
                                                      self.physical_sizes[0])
        sphere = Topography.make_sphere(self.radius, res, size)
        # here, i deliberately avoid using the make_system, because I want to
        # explicitly test the dumb (yet safer) way of computing problems with a
        # free, non-periodic  boundary. A user who invokes a system constructor
        # directliy like this is almost certainly mistaken
        S = SmoothContactSystem(substrate, self.smooth, sphere)
        offset = -self.sig
        disp = np.zeros(substrate.nb_domain_grid_pts)

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
        ref_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.nc')
        out_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.out')
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
             "reference: PyCo: {}, pycontact: {}, error: {}, tol: "
            "{}").format(potential.r_c, ref_data.lj_rc2, error, tol))
        ## 2. replicate substrate
        res = (ref_data.size//2, ref_data.size//2)

        size= tuple((float(r) for r in res))
        young = 2. # pycontact convention (hardcoded)
        substrate = Solid.FreeFFTElasticHalfSpace(res, young, size)

        ## 3. replicate surface
        radius = ref_data.Hertz
        centre = (15.5, 15.5)
        surface = Topography.make_sphere(radius, res, size, centre=centre)
        ## ref_h = -np.array(ref_data.variables['h'])
        ## ref_h -= ref_h.max()
        ## surface = Topography.NumpySurface(ref_h)
        ## 4. Set up system:
        S = SmoothContactSystem(substrate, potential, surface)

        ref_profile = np.array(
            ref_data.variables['h']+ref_data.variables['avgh'][0])[:32, :32]
        offset = -.8*potential.r_c
        gap = S.compute_gap(np.zeros(substrate.nb_domain_grid_pts), offset)
        diff = ref_profile-gap
        # pycontact centres spheres at (n + 0.5, m + 0.5). need to correct for test
        correction = radius - np.sqrt(radius**2-.5)
        error = Tools.mean_err(ref_profile + correction, gap)
        self.assertTrue(
            error < tol,
            ("initial profiles differ (mean error ē = {} > tol = {}, mean gap = {}"
            "mean ref_profile = {})").format(
                error, tol, gap.mean(), (ref_profile + correction).mean()))
        # pycontact does not save the offset in the nc, so this one has to be
        # taken on faith
        fun = S.objective(offset + correction, gradient=True)
        fun_hard = S.objective(offset + correction, gradient=False)

        ## initial guess (cheating) is the solution of pycontact
        disp = np.zeros(S.substrate.nb_domain_grid_pts)
        disp[:ref_data.size, :ref_data.size] = -ref_data.variables['u'][0]
        gap = S.compute_gap(disp, offset)
        print("gap:     min, max = {}, offset = {}".format((gap.min(), gap.max()), offset))
        print("profile: min, max = {}".format((S.surface.heights().min(), S.surface.heights().max())))
        options = dict(ftol = 1e-15, gtol = 1e-12)
        result = minimize(fun, disp, jac=True, callback=S.callback(force=True), method = 'L-BFGS-B', options=options)

        # options = dict(ftol = 1e-12, gtol = 1e-10, maxiter=100000)
        # result = minimize(fun_hard, disp, jac=False, callback=S.callback(force=False), method = 'L-BFGS-B', options=options)

        e, force = fun(result.x)



        error = abs(ref_N - S.compute_normal_force())
        # here the answers differ slightly, relaxing the tol for this one
        ftol = 1e-7

        ## import matplotlib.pyplot as plt
        ## fig = plt.figure()
        ## CS = plt.contourf(ref_data.variables['f'][0])
        ## plt.colorbar(CS)
        ## plt.title("ref")
        ## fig = plt.figure()
        ## CS = plt.contourf(S.substrate.force[:32, :32])
        ## plt.colorbar(CS)
        ## plt.title("substrate")
        ## fig = plt.figure()
        ## CS = plt.contourf(S.interaction.force[:32, :32])
        ## plt.colorbar(CS)
        ## plt.title("interaction")
        ## plt.show()

        ## fig = plt.figure()
        ## CS = plt.contourf(-ref_data.variables['u'][0][:32, :32])
        ## plt.colorbar(CS)
        ## plt.title("ref_u")
        ## fig = plt.figure()
        ## CS = plt.contourf(result.x.reshape([64, 64])[:32, :32])
        ## plt.colorbar(CS)
        ## plt.title("my_u")
        ## fig = plt.figure()
        ## CS = plt.contourf(result.x.reshape([64, 64])[:32, :32] + ref_data.variables['u'][0][:32, :32])
        ## plt.colorbar(CS)
        ## plt.title("my_u - ref_u")
        ## plt.show()

        self.assertTrue(
            error < ftol,
            ("resulting normal forces differ: error = {} > tol = {}, "
             "ref_force_n = {}, my_force_n = {}\nOptimResult was {}\nelast energy = {}\ninteraction_force = {}\nsubstrate_force = {}\n System type = '{}'").format(
                 error, ftol, ref_N, S.compute_normal_force(), result,
                 S.substrate.energy, S.interaction.force.sum(),
                 S.substrate.force.sum(), type(S)))
        error = Tools.mean_err(
            disp, result.x.reshape(S.substrate.nb_domain_grid_pts))
        self.assertTrue(
            error < ftol,
            "resulting displacements differ: error = {} > tol = {}".format(
                error, ftol))

    def test_size_insensitivity(self):
        tol = 1e-6
        ref_fpath = os.path.join(BASEDIR,'ref_smooth_sphere.nc')
        out_fpath = os.path.join(BASEDIR, 'ref_smooth_sphere.out')
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
             "reference: PyCo: {}, pycontact: {}, error: {}, tol: "
            "{}").format(potential.r_c, ref_data.lj_rc2, error, tol))
        nb_compars = 3
        normalforce = np.zeros(nb_compars)
        options = dict(ftol = 1e-12, gtol = 1e-10)

        for i, nb_grid_pts in ((i, ref_data.size//4*2**i) for i in range(nb_compars)):
            res = (nb_grid_pts, nb_grid_pts)

            size= tuple((float(r) for r in res))
            young = 2. # pycontact convention (hardcoded)
            substrate = Solid.FreeFFTElasticHalfSpace(res, young, size)

            ## 3. replicate surface
            radius = ref_data.Hertz
            surface = Topography.make_sphere(radius, res, size)

            ## 4. Set up system:
            S = SmoothContactSystem(substrate, potential, surface)
            # pycontact does not save the offset in the nc, so this one has to be
            # taken on faith
            offset = -.8*potential.r_c
            fun = S.objective(offset, gradient=True)
            disp = np.zeros(np.prod(res)*4)
            result = minimize(fun, disp, jac=True,
                              method = 'L-BFGS-B', options=options)
            normalforce[i] = S.interaction.force.sum()
            error = abs(normalforce-normalforce.mean()).mean()
        self.assertTrue(error < tol, "error = {:.15g} > tol = {}, N = {}".format(
            error, tol, normalforce))


