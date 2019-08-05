#
# Copyright 2018-2019 Antoine Sanner
#           2018-2019 Lars Pastewka
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
tests that the Fast optimization for free (non-periodic) systems is
consistent with computing the full system
"""

import time
import unittest

import numpy as np
import pytest
from scipy.optimize import minimize

from NuMPI import MPI

import PyCo.SolidMechanics as Solid
import PyCo.ContactMechanics as Contact
import PyCo.Topography as Topography
import PyCo.Tools as Tools
from PyCo.System.Systems import SmoothContactSystem, NonSmoothContactSystem
from PyCo.System.SmoothSystemSpecialisations import FastSmoothContactSystem
from PyCo.System import make_system


pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")

#class SmoothSystemTest(unittest.TestCase):
import pytest

@pytest.mark.parametrize("r_c",[2., 6.])
@pytest.mark.parametrize("young",[3., 10, 100.]) # ,10.,100.
def test_minimization_simplesmoothmin(young, r_c):


    eps=1.
    sig=1.#2


    radius=4.

    base_res = 32
    res = (base_res, base_res)

    size= (15.,15.)
    surface = Topography.make_sphere(radius, res, size,
                                     standoff=float("inf"))
    ext_surface = Topography.make_sphere(radius, [2 * r for r in res], [2 * s for s in size],
                                 centre=[ s/2 for s in size], standoff=float('inf'))

    substrate = Solid.FreeFFTElasticHalfSpace(
        res, young, size, fft="numpy")

    pot = Contact.LJ93SimpleSmoothMin(eps, sig, r_c=r_c, r_ti=0.5)

    if False:
        import matplotlib.pyplot as plt
        fig,(axp, axf) = plt.subplots(1,2)

        z = np.linspace(0.5,10,50)

        v,dv, ddv = pot.evaluate(z, True, True)
        axp.plot(z,v)
        axf.plot(z,dv)

    S = SmoothContactSystem(substrate,
                            pot,
                            surface)

    #print("testing: {}, rc = {}, offset= {}".format(pot.__class__, S.interaction.r_c, S.interaction.offset))
    offset = .8 * S.interaction.r_c
    fun = S.objective(offset, gradient=True)

    options = dict(ftol=1e-18, gtol=1e-10)
    disp = S.shape_minimisation_input(
        np.zeros(substrate.nb_domain_grid_pts))

    lbounds  =S.shape_minimisation_input(ext_surface.heights() + offset)
    bnds = tuple(zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
    result = minimize(fun, disp, jac=True,
                      method='L-BFGS-B', options=options)#, bounds=bnds)
    if False:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()

        #ax.pcolormesh(result.x.reshape(substrate.computational_nb_grid_pts))
        ax.pcolormesh(S.interaction.force)
    #    np.savetxt("{}_forces.txt".format(pot_class.__name__), S.interaction.force)
        ax.set_xlabel("x")
        ax.set_ylabel("")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        plt.show(block=True)
    assert result.success, "{}".format(result)

@pytest.mark.skip(reason="Strange problem see issue #139 work on this in the mufft branch")
@pytest.mark.parametrize("base_res",[pytest.param(128, marks=pytest.mark.xfail),
                                     256])
@pytest.mark.parametrize("young", [3.,100.]) # mit young = 100 geht auch LJ93smoothMin durch
@pytest.mark.parametrize("pot_class",[pytest.param(Contact.LJ93smooth, marks=pytest.mark.xfail),
                                      Contact.LJ93smoothMin])
def test_minimization(pot_class, young, base_res):

    eps=1
    sig=2
    gam=5

    radius= 4.

    res = (base_res, base_res)

    size= (15.,15.)
    surface = Topography.make_sphere(radius, res, size,
                                     standoff=float("inf"))
    ext_surface = Topography.make_sphere(radius, [2 * r for r in res], [2 * s for s in size],
                                 centre=[ s/2 for s in size], standoff=float("inf"))

    substrate = Solid.FreeFFTElasticHalfSpace(res, young, size, fft="numpy")

    if pot_class==Contact.VDW82smoothMin:
        pot = pot_class(gam * eps **8 / 3, 16 * np.pi * gam * eps**2,gamma=gam)
    elif pot_class== Contact.LJ93SimpleSmoothMin :
        pot = pot_class(eps, sig, r_c = 10., r_ti=0.5)
    else:
        pot = pot_class(eps, sig, gam, )
    if hasattr(pot, "r_ti"):
        assert pot.r_ti < pot.r_t

    if False:
        import matplotlib.pyplot as plt
        fig,(axp, axf) = plt.subplots(1,2)

        z = np.linspace(0.5,10,50)

        v,dv, ddv = pot.evaluate(z, True, True)
        axp.plot(z,v)
        axf.plot(z,dv)

    S = SmoothContactSystem(substrate,
                            pot,
                            surface)

    print("testing: {}, rc = {}, offset= {}".format(pot.__class__, S.interaction.r_c, S.interaction.offset))
    offset = .8 * S.interaction.r_c
    fun = S.objective(offset, gradient=True)

    options = dict(ftol=1e-16, gtol=1e-8)
    disp = S.shape_minimisation_input(
        np.zeros(substrate.nb_domain_grid_pts))

    lbounds  =S.shape_minimisation_input(ext_surface.heights() + offset)
    bnds = tuple(zip(lbounds.tolist(), [None for i in range(len(lbounds))]))
    result = minimize(fun, disp, jac=True,
                      method='L-BFGS-B', options=options)#, bounds=bnds)
    if False:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()

        #ax.pcolormesh(result.x.reshape(substrate.computational_nb_grid_pts))
        ax.pcolormesh(S.interaction.force)
        np.savetxt("{}_forces.txt".format(pot_class.__name__), S.interaction.force)
        ax.set_xlabel("x")
        ax.set_ylabel("")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        plt.show(block=True)
    assert result.success, "{}".format(result)


class FastSystemTest(unittest.TestCase):
    def setUp(self):
        self.physical_sizes = (15, 15)#(7.5+5*rand(), 7.5+5*rand())
        self.radius = 4
        base_res = 64 # TODO: put this back on 32, see issue #139
        self.res = (base_res, base_res)
        self.young = 3#+2*random()

        self.substrate = Solid.FreeFFTElasticHalfSpace(
            self.res, self.young, self.physical_sizes, fft="serial")

        self.eps = 1# +np.random.rand()
        self.sig = 2# +np.random.rand()
        self.gam = 5# +np.random.rand()
        self.rcut = 2.5*self.sig# +np.random.rand()
        self.interaction = Contact.LJ93smooth(self.eps, self.sig, self.gam)
        #self.min_pot = Contact.LJ93smooth(self.eps, self.sig, self.gam)
        self.min_pot = Contact.LJ93smoothMin(self.eps, self.sig, self.gam)
        #self.interaction = Contact.LJ93(self.eps, self.sig)
        #self.min_pot = Contact.LJ93SimpleSmooth(self.eps, self.sig, self.rcut)

        #self.interaction =Contact.ExpPotential(self.gam, 0.05, self.rcut)
        #self.min_pot = Contact.ExpPotential(self.gam, 0.05, self.rcut)

        self.surface = Topography.make_sphere(self.radius, self.res,
                                              self.physical_sizes,
                                              standoff=float('inf'))

        if False:
            import matplotlib.pyplot as plt
            fig, (axE, axF, axC) = plt.subplots(3, 1)
            z = np.linspace(0.8, 2, 100)
            V, dV, ddV = self.min_pot.evaluate(z, True, True, True)
            axE.plot(z, V)
            axF.plot(z, dV)
            axC.plot(z, ddV)

            V, dV, ddV = self.interaction.evaluate(z, True, True, True)
            axE.plot(z, V)
            axF.plot(z, dV)
            axC.plot(z, ddV)

            plt.show(block=True)



    def test_FastSmoothContactSystem(self):
        S = FastSmoothContactSystem(self.substrate,
                                    self.interaction,
                                    self.surface)
        fun = S.objective(.95*self.interaction.r_c)
        print(fun(np.zeros(S.babushka.substrate.nb_domain_grid_pts)))

    def test_SystemFactory(self):
        S = make_system(self.substrate,
                        self.interaction,
                        self.surface)
        print("Mofo is periodic ?: ", self.substrate.is_periodic())
        print("substrate: ", self.substrate)
        self.assertIsInstance(S, FastSmoothContactSystem)
        self.assertIsInstance(S, SmoothContactSystem)

    def test_babushka_translations(self):
        S = FastSmoothContactSystem(self.substrate,
                                    self.interaction,
                                    self.surface)
        fun = S.objective(.95*self.interaction.r_c)


    def test_equivalence(self):
        tol = 1e-6
        # here, i deliberately avoid using the make_system, because I want to
        # explicitly test the dumb (yet safer) way of computing problems with a
        # free, non-periodic  boundary. A user who invokes a system constructor
        # directliy like this is almost certainly mistaken
        systems = (SmoothContactSystem, FastSmoothContactSystem)
        def eval(system):
            print("running for system {}".format(system.__name__))
            S = system(self.substrate,
                       self.min_pot,
                       self.surface)
            offset = .8 * S.interaction.r_c
            fun = S.objective(offset, gradient=True)

            options = dict(ftol = 1e-18, gtol = 1e-10)
            disp = S.shape_minimisation_input(
                np.zeros(self.substrate.nb_domain_grid_pts))
            bla = fun(disp)
            result = minimize(fun, disp, jac=True,
                              method = 'L-BFGS-B', options=options)
            assert result.success, "{}".format(result)
            if system.is_proxy():
                dummy, force, disp = S.deproxified()

            else:
                disp = S.shape_minimisation_output(result.x)
            gap = S.compute_gap(disp, offset)
            gap[np.isinf(gap)] = self.min_pot.r_c

            print('r_min = {}'.format(self.min_pot.r_min))
            return S.interaction.force, disp, gap, S.compute_normal_force()

        def timer(fun, *args):
            start = time.perf_counter()
            res = fun(*args)
            delay = time.perf_counter()-start
            return res, delay

        (((force_slow, disp_slow, gap_slow, N_slow), slow_time),
         ((force_fast, disp_fast, gap_fast, N_fast), fast_time)) = tuple(
             (timer(eval, system) for system in systems))
        error = Tools.mean_err(disp_slow, disp_fast)

        print("Normal forces: fast: {}, slow: {}, error: {}".format(
            N_fast, N_slow, abs(N_slow- N_fast)))

        print("timings: fast: {}, slow: {}, gain: {:2f}%".format(
            fast_time, slow_time, 100*(1-fast_time/slow_time)))
        self.assertTrue(error < tol,
                        "error = {} > tol = {}".format(
                            error, tol))

    def test_minimize_proxy(self):
        tol = 1e-6
        # here, i deliberately avoid using the make_system, because I want to
        # explicitly test the dumb (yet safer) way of computing problems with a
        # free, non-periodic  boundary. A user who invokes a system constructor
        # directliy like this is almost certainly mistaken
        systems = (SmoothContactSystem, FastSmoothContactSystem)
        def eval(system):
            print("running for system {}".format(system.__name__))
            S = system(self.substrate,
                       self.min_pot,
                       self.surface)
            offset = .8 * S.interaction.r_c
            options = dict(ftol = 1e-18, gtol = 1e-10)
            result = S.minimize_proxy(offset, options=options)

            gap = S.compute_gap(S.disp, offset)
            gap[np.isinf(gap)] = self.min_pot.r_c

            return S.interaction.force, S.disp, gap, S.compute_normal_force()

        def timer(fun, *args):
            start = time.perf_counter()
            res = fun(*args)
            delay = time.perf_counter()-start
            return res, delay

        (((force_slow, disp_slow, gap_slow, N_slow), slow_time),
         ((force_fast, disp_fast, gap_fast, N_fast), fast_time)) = tuple(
             (timer(eval, system) for system in systems))
        error = Tools.mean_err(disp_slow, disp_fast)

        print("Normal forces: fast: {}, slow: {}, error: {}".format(
            N_fast, N_slow, abs(N_slow- N_fast)))

        print("timings: fast: {}, slow: {}, gain: {:2f}%".format(
            fast_time, slow_time, 100*(1-fast_time/slow_time)))
        self.assertTrue(error < tol,
                        "error = {} > tol = {}".format(
                            error, tol))

    def test_babuschka_eval(self):
        tol = 1e-6
        # here, i deliberately avoid using the make_system, because I want to
        # explicitly test the dumb (yet safer) way of computing problems with a
        # free, non-periodic  boundary. A user who invokes a system constructor
        # directliy like this is almost certainly mistaken
        S = FastSmoothContactSystem(self.substrate,
                                    self.min_pot,
                                    self.surface)
        offset = .8 * S.interaction.r_c
        S.create_babushka(offset)
        S.babushka.evaluate(
            np.zeros(S.babushka.substrate.nb_domain_grid_pts), offset,
            forces=True)
        F_n = S.babushka.compute_normal_force()
        babushka = S.babushka
        S = SmoothContactSystem(self.substrate,
                                self.min_pot,
                                self.surface)
        S.evaluate(np.zeros(S.substrate.nb_domain_grid_pts), offset, forces=True)
        F_n2 = S.compute_normal_force()

        error = abs(1 - F_n/F_n2)
        tol = 1e-14
        self.assertTrue(error < tol,
                        ("F_n = {}, F_n2 = {}, should be equal. type(S) = {}. "
                         "type(S.babushka) = {}, err = {}").format(
                             F_n, F_n2, type(S), type(babushka), error))

    def test_unit_neutrality(self):
        tol = 2e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c   = 1. +9# np.random.rand()
        force_c    = 2. + 1#np.random.rand()
        pressure_c = force_c/length_c**2
        energy_per_area_c   = force_c/length_c
        energy_c   = force_c*length_c

        young = (self.young, pressure_c*self.young)
        size = (self.physical_sizes, tuple((length_c * s for s in self.physical_sizes)))
        print("SIZES!!!!! = ", size)
        radius = (self.radius, length_c*self.radius)
        res = self.res
        eps = (self.eps, energy_per_area_c*self.eps)
        sig = (self.sig, length_c*self.sig)
        gam = (self.gam, energy_per_area_c*self.gam)

        systems = list()
        offsets = list()
        length_rc = (1., 1./length_c)
        force_rc = (1., 1./force_c)
        energy_per_area_rc = (1., 1./energy_per_area_c)
        energy_rc = (1., 1./energy_c)

        for i in range(2):
            substrate = Solid.PeriodicFFTElasticHalfSpace(res, young[i],
                                                          size[i])
            interaction = Contact.LJ93smoothMin(
                eps[i], sig[i], gam[i])
            surface = Topography.make_sphere(radius[i], res, size[i], standoff=float(sig[i]*1000))
            systems.append(make_system(substrate, interaction, surface))
            offsets.append(.8*systems[i].interaction.r_c)

        gaps = list()
        for i in range(2):
            gap = systems[i].compute_gap(np.zeros(systems[i].nb_grid_pts), offsets[i])
            gaps.append(gap*length_rc[i])

        error = Tools.mean_err(gaps[0], gaps[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

        forces = list()
        for i in range(2):
            energy, force = systems[i].evaluate(np.zeros(res), offsets[i], forces=True)
            forces.append(force*force_rc[i])

        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

        energies, forces = list(), list()
        substrate_energies = list()
        interaction_energies = list()
        disp = np.random.random(res)
        disp -= disp.mean()
        disp = (disp, disp*length_c)
        gaps = list()

        for i in range(2):
            energy, force = systems[i].evaluate(disp[i], offsets[i], forces=True)
            gap = systems[i].compute_gap(disp[i], offsets[i])
            gaps.append(gap*length_rc[i])
            energies.append(energy*energy_rc[i])
            substrate_energies.append(systems[i].substrate.energy*energy_rc[i])
            interaction_energies.append(systems[i].interaction.energy*energy_rc[i])
            forces.append(force*force_rc[i])

        error = Tools.mean_err(gaps[0], gaps[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

        error = Tools.mean_err(forces[0], forces[1])

        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

        error = abs(interaction_energies[0] - interaction_energies[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))


        error = abs(substrate_energies[0] - substrate_energies[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}, (c = {})".format(error, tol, energy_c))

        error = abs(energies[0] - energies[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

        disps = list()
        for i in range(2):
            options = dict(ftol = 1e-32, gtol = 1e-20)
            result = systems[i].minimize_proxy(offsets[i], options=options)
            disps.append(systems[i].shape_minimisation_output(result.x)*length_rc[i])

        error = Tools.mean_err(disps[0], disps[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}, (c = {})".format(error, tol, length_c))


    def test_BabushkaBoundaryError(self):
        """
        makes a Simulation in JKR-Like condition so that the contact area jump (snap-in) will lead to a too small
        babushka-subdomain area
        """
        with self.assertRaises(FastSmoothContactSystem.BabushkaBoundaryError):
            s = 128
            n = 64
            dx = 2
            size = (s, s)
            res = (n,n)
            radius = 100
            young = 1
            gam = 0.05

            surface = Topography.make_sphere(radius, res, size)
            ext_surface = Topography.make_sphere(radius, (2 * n, 2 * n), (2 * s, 2 * s), centre=(s / 2, s / 2))

            interaction = Contact.LJ93smoothMin(young/18*np.sqrt(2/5),2.5**(1/6),gamma=gam)

            substrate = Solid.FreeFFTElasticHalfSpace(surface.nb_grid_pts, young, surface.physical_sizes)
            system = FastSmoothContactSystem(substrate, interaction, surface, margin=4)

            start_disp = - interaction.r_c + 1e-10
            load_history = np.concatenate((
                np.array((start_disp,)),
                np.arange(-1.63, -1.6, 2e-3),
                np.arange(-1.6, 0.6, 2e-1)[1:]))

            u=None
            for offset in load_history:

                opt = system.minimize_proxy(offset,
                                            u,
                                            method='L-BFGS-B',
                                            options=dict(ftol=1e-18, gtol=1e-10),
                                            lbounds=ext_surface.heights() + offset)

                u = system.disp

            import matplotlib.pyplot as plt
            X, Y = np.meshgrid((np.arange(0, int(n / 2))) * dx, (np.arange(0, int(n / 2))) * dx)
            fig, ax = plt.subplots()
            plt.colorbar(ax.pcolormesh(X, Y, substrate.interact_forces[-1, int(n / 2):, int(n / 2):]))


    def test_FreeBoundaryError(self):
        """
        Returns
        -------
        """
        radius = 100
        young = 1

        s = 128.
        n = 64
        dx = s/n
        res = (n, n)
        size = (s, s)

        centre = (0.75*s, 0.5* s)

        topography = Topography.make_sphere(radius, res, size,centre=centre)
        ext_topography = Topography.make_sphere(radius, (2 * n, 2 * n), (2 * s, 2 * s), centre=centre)

        substrate = Solid.FreeFFTElasticHalfSpace(topography.nb_grid_pts, young,
                                                  topography.physical_sizes)

        for system in [NonSmoothContactSystem(substrate, Contact.HardWall(), topography),
                       SmoothContactSystem(substrate, Contact.LJ93SimpleSmooth(0.01,0.01,10), topography)]:
            with self.subTest(system=system):
                offset = 15
                with self.assertRaises(Solid.FreeFFTElasticHalfSpace.FreeBoundaryError):
                    opt = system.minimize_proxy(offset=offset)
                if False:
                    import matplotlib.pyplot as plt
                    X, Y = np.meshgrid((np.arange(0, n)) * dx,
                                       (np.arange(0, n)) * dx)
                    fig, ax = plt.subplots()
                    plt.colorbar(
                        ax.pcolormesh(X, Y, substrate.force)
                    )
                    plt.show(block=True)

