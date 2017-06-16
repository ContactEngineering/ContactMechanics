#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   03-FFTElasticHalfSpaceTest.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests the fft elastic halfspace implementation

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
    from numpy.linalg import norm
    from numpy.random import rand, random
    from numpy.fft import rfftn, irfftn
    import time

    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    import PyCo.Tools as Tools
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class PeriodicFFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()
        self.poisson = 0.23

    def test_consistency(self):
        pressure = list()
        base_res = 128
        tol = 1e-4
        for i in (1, 2):
            s_res = base_res*i
            test_res = (s_res, s_res)
            hs = PeriodicFFTElasticHalfSpace(test_res, self.young, self.size)
            forces = np.zeros(test_res)
            forces[:s_res//2,:s_res//2] = 1.

            pressure.append(hs.evaluate_disp(forces)[::i,::i]*hs.area_per_pt)
        error = ((pressure[0]-pressure[1])**2).sum().sum()/base_res**2
        self.assertTrue(error < tol, "error = {}".format(error))


    def test_parabolic_shape_force(self):
        """ tests whether the Elastic energy is a quadratic function of the
            applied force"""
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size)
        force = random(self.res)
        force -= force.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            disp = hs.evaluate_disp(i*force)
            El[i] = hs.evaluate_elastic_energy(i*force, disp)
        tol = 1e-10
        error = norm(El/El[1]-np.arange(nb_tests)**2)
        self.assertTrue(error<tol)

    def test_parabolic_shape_disp(self):
        """ tests whether the Elastic energy is a quadratic function of the
            applied displacement"""
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size)
        disp = random(self.res)
        disp -= disp.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            force = hs.evaluate_force(i*disp)
            El[i] = hs.evaluate_elastic_energy(i*force, disp)
        tol = 1e-10
        error = norm(El/El[1]-np.arange(nb_tests)**2)
        self.assertTrue(error<tol)

    def test_gradient(self):
        res = size = (2, 2)
        disp = random(res)
        disp -= disp.mean()
        hs = PeriodicFFTElasticHalfSpace(res, self.young, size)
        hs.compute(disp, forces = True)
        f =  hs.energy
        g = -hs.force
        approx_g = Tools.evaluate_gradient(
            lambda x: hs.evaluate(x, forces=True)[0], disp, 1e-5)

        tol = 1e-8
        error = Tools.mean_err(g, approx_g)
        msg = []
        msg.append("f = {}".format(f))
        msg.append("g = {}".format(g))
        msg.append('approx = {}'.format(approx_g))
        msg.append("error = {}".format(error))
        msg.append("tol = {}".format(tol))
        self.assertTrue(error < tol, ", ".join(msg))


    def test_force_disp_reversibility(self):
        ## since only the zero-frequency is rejected, any force/disp field with
        ## zero mean should be fully reversible
        tol = 1e-10
        for res in ((self.res[0],), self.res, (self.res[0]+1, self.res[1]),
                    (self.res[0], self.res[1]+1)):
            hs = PeriodicFFTElasticHalfSpace(res, self.young, self.size)
            disp = random(res)
            disp -= disp.mean()

            error = Tools.mean_err(disp, hs.evaluate_disp(hs.evaluate_force(disp)))
            self.assertTrue(
                error < tol,
                "for resolution = {}, error = {} > tol = {}".format(
                    res, error, tol))

            force = random(res)
            force -= force.mean()

            error = Tools.mean_err(force, hs.evaluate_force(hs.evaluate_disp(force)))
            self.assertTrue(
                error < tol,
                "for resolution = {}, error = {} > tol = {}".format(
                    res, error, tol))

    def test_energy(self):
        tol = 1e-10
        l = 2 + rand() # domain length
        a = 3 + rand() # amplitude of force
        E = 4 + rand() # Young's Mod
        for res in [4, 8, 16]:
            area_per_pt = l/res
            x = np.arange(res)*l/res
            force = a*np.cos(2*np.pi/l*x)

            ## theoretical FFT of force
            Fforce = np.zeros_like(x)
            Fforce[1] = Fforce[-1] = res/2.*a

            ## theoretical FFT of disp
            Fdisp = np.zeros_like(x)
            Fdisp[1] = Fdisp[-1] = res/2.*a/E*l/(2*np.pi)

            ## verify consistency
            hs = PeriodicFFTElasticHalfSpace(res, E, l)
            fforce = rfftn(force)
            fdisp = hs.weights*fforce
            self.assertTrue(
                Tools.mean_err(fforce, Fforce, rfft=True)<tol, "fforce = \n{},\nFforce = \n{}".format(
                    fforce.real, Fforce))
            self.assertTrue(
                Tools.mean_err(fdisp, Fdisp, rfft=True)<tol, "fdisp = \n{},\nFdisp = \n{}".format(
                    fdisp.real, Fdisp))

            ##Fourier energy
            E = .5*np.dot(Fforce/area_per_pt, Fdisp)/res

            disp = hs.evaluate_disp(force)
            e =  hs.evaluate_elastic_energy(force, disp)
            kdisp = hs.evaluate_k_disp(force)
            ee = hs.evaluate_elastic_energy_k_space(fforce, kdisp)
            self.assertTrue(
                abs(e-ee) < tol,
                 "violate Parseval: e = {}, ee = {}, ee/e = {}".format(
                    e, ee, ee/e))

            self.assertTrue(
                abs(E-e)<tol,
                "theoretical E = {}, computed e = {}, diff(tol) = {}({})".format(
                    E, e, E-e, tol))

    def test_unit_neutrality(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c   = 1. + np.random.rand()
        force_c    = 1. + np.random.rand()
        pressure_c = force_c/length_c**2
        energy_c   = force_c*length_c

        length_rc = (1., 1./length_c)
        force_rc = (1., 1./force_c)
        pressure_rc = (1., 1./pressure_c)
        energy_rc = (1., 1./energy_c)
        resolution = (32, 32)
        young = (self.young, pressure_c*self.young)
        size = self.size[0], 2*self.size[1]
        size = (size, tuple((length_c*s for s in size)))
        print('SELF.SIZE = {}'.format(self.size))

        disp = np.random.random(resolution)
        disp -= disp.mean()
        disp = (disp, disp*length_c)

        forces = list()
        for i in range(2):
            sub = PeriodicFFTElasticHalfSpace(resolution, young[i], size[i])
            force = sub.evaluate_force(disp[i])
            forces.append(force*force_rc[i])
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

    def test_unit_neutrality1D(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c   = 1. + np.random.rand()
        force_c    = 2. + np.random.rand()
        pressure_c = force_c/length_c**2
        energy_c   = force_c*length_c
        force_per_length_c = force_c/length_c

        length_rc = (1., 1./length_c)
        force_rc = (1., 1./force_c)
        pressure_rc = (1., 1./pressure_c)
        energy_rc = (1., 1./energy_c)
        force_per_length_rc = (1., 1./force_per_length_c)

        resolution = (32, )
        young = (self.young, pressure_c*self.young)
        size = (self.size[0], length_c*self.size[0])

        disp = np.random.random(resolution)
        disp -= disp.mean()
        disp = (disp, disp*length_c)

        forces = list()
        subs = tuple((PeriodicFFTElasticHalfSpace(resolution, y, s) for y, s in
                      zip(young, size)))
        forces = tuple((s.evaluate_force(d)*f_p_l for s, d, f_p_l in
                        zip(subs, disp, force_per_length_rc)))
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

    def test_uniform_displacement(self):
        """ tests whether uniform displacement returns stiffness_q0"""
        sq0 = 1.43
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size,
                                         stiffness_q0=sq0)
        force = hs.evaluate_force(-np.ones(self.res))
        self.assertAlmostEqual(force.sum()/np.prod(self.size), sq0)

    def test_uniform_displacement_finite_height(self):
        """ tests whether uniform displacement returns stiffness_q0"""
        h0 = 3.45
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size,
                                         poisson=self.poisson, thickness=h0)
        force = hs.evaluate_force(-np.ones(self.res))
        M = (1-self.poisson)/((1-2*self.poisson)*(1+self.poisson))*self.young
        self.assertAlmostEqual(force.sum()/np.prod(self.size), M/h0)

class FreeFFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3+2*random()

    def test_consistency(self):
        pressure = list()
        base_res = 32
        tol = 1e-1
        for i in (1, 2):
            s_res = base_res*i
            test_res = (s_res, s_res)
            hs = FreeFFTElasticHalfSpace(test_res, self.young, self.size)
            forces = np.zeros([2*r for r in test_res])
            forces[:s_res//2,:s_res//2] = 1.

            pressure.append(hs.evaluate_disp(forces)[::i,::i]*hs.area_per_pt)
        error = ((pressure[0]-pressure[1])**2).sum().sum()/base_res**2
        self.assertTrue(error < tol, "error = {}, tol = {}".format(error, tol))

    def test_FourierCoeffCost(self):
        print()
        print('Computation of Fourier coefficients:')
        for i in range(1, 4):
            res = (2**i, 2**i)
            hs = FreeFFTElasticHalfSpace(res, self.young, self.size)

            start = time.perf_counter()
            w2, f2 =hs._compute_fourier_coeffs2()
            duration2 = time.perf_counter()-start

            start = time.perf_counter()
            w3, f3 =hs._compute_fourier_coeffs()
            duration3 = time.perf_counter()-start

            print(
                "for {0[0]}: np {1:.2f}, mat_scipy {2:.2f} ms({3:.1f}%)".format(
                    res, duration2*1e3, duration3*1e3, 1e2*(1-duration3/duration2)))
            error = Tools.mean_err(w2, w3)
            self.assertTrue(error == 0)

    def test_realnessEnergy(self):
        hs = FreeFFTElasticHalfSpace(self.res, self.young, self.size)
        force = np.zeros(hs.computational_resolution)
        force[:self.res[0], :self.res[1]] = np.random.random(self.res)
        force[:self.res[0], :self.res[1]] -= force[:self.res[0], :self.res[1]].mean()
        kdisp = hs.evaluate_k_disp(force)
        kforce = rfftn(force)
        np_pts = np.prod(self.res)
        area_per_pt = np.prod(self.size)/np_pts
        energy = .5*(np.vdot(-kforce, kdisp)+
                     np.vdot(-kforce[..., :-1], kdisp[..., :-1]))/np_pts
        error = abs(energy.imag)
        tol = 1e-10
        self.assertTrue(error < tol,
                         "error (imaginary part) = {}, tol = {}".format(
                             error, tol))
        error = abs(energy-hs.evaluate_elastic_energy_k_space(kforce, kdisp))
        self.assertTrue(error < tol,
                        ("error (comparison) = {}, tol = {}, energy = {}, "
                         "kenergy = {}").format(
                             error, tol, energy,
                             hs.evaluate_elastic_energy_k_space(kforce, kdisp)))

    def test_energy(self):
        tol = 1e-10
        l = 2 + rand() # domain length
        a = 3 + rand() # amplitude of force
        E = 4 + rand() # Young's Mod
        for res in [4, 8, 16]:
            area_per_pt = l/res
            x = np.arange(res)*area_per_pt
            force = a*np.cos(2*np.pi/l*x)

            ## theoretical FFT of force
            Fforce = np.zeros_like(x)
            Fforce[1] = Fforce[-1] = res/2.*a

            ## theoretical FFT of disp
            Fdisp = np.zeros_like(x)
            Fdisp[1] = Fdisp[-1] = res/2.*a/E*l/(2.*np.pi)

            ## verify consistency
            hs = PeriodicFFTElasticHalfSpace(res, E, l)
            fforce = rfftn(force)
            fdisp = hs.weights*fforce
            self.assertTrue(
                Tools.mean_err(fforce, Fforce, rfft=True)<tol, "fforce = \n{},\nFforce = \n{}".format(
                    fforce.real, Fforce))
            self.assertTrue(
                Tools.mean_err(fdisp, Fdisp, rfft=True)<tol, "fdisp = \n{},\nFdisp = \n{}".format(
                    fdisp.real, Fdisp))

            ##Fourier energy
            E = .5*np.dot(Fforce/area_per_pt, Fdisp)/res

            disp = hs.evaluate_disp(force)
            e =  hs.evaluate_elastic_energy(force, disp)
            kdisp = hs.evaluate_k_disp(force)
            self.assertTrue(abs(disp - irfftn(kdisp)).sum()<tol,
                            ("disp   = {}\n"
                             "ikdisp = {}").format(disp, irfftn(kdisp)))
            ee = hs.evaluate_elastic_energy_k_space(fforce, kdisp)
            self.assertTrue(
                abs(e-ee) < tol,
                 "violate Parseval: e = {}, ee = {}, ee/e = {}".format(
                    e, ee, ee/e))

            self.assertTrue(
                abs(E-e)<tol,
                "theoretical E = {}, computed e = {}, diff(tol) = {}({})".format(
                    E, e, E-e, tol))

    def test_unit_neutrality(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c   = 1. + np.random.rand()
        force_c    = 2. + np.random.rand()
        pressure_c = force_c/length_c**2
        energy_c   = force_c*length_c

        length_rc = (1., 1./length_c)
        force_rc = (1., 1./force_c)
        pressure_rc = (1., 1./pressure_c)
        energy_rc = (1., 1./energy_c)
        resolution = (32, 32)
        young = (self.young, pressure_c*self.young)
        size = (self.size, tuple((length_c*s for s in self.size)))

        comp_resolution = tuple((2*res for res in resolution))
        disp = np.random.random(comp_resolution)
        disp -= disp.mean()
        disp = (disp, disp*length_c)

        forces = list()
        for i in range(2):
            sub = FreeFFTElasticHalfSpace(resolution, young[i], size[i])
            force = sub.evaluate_force(disp[i])
            forces.append(force*force_rc[i])
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

