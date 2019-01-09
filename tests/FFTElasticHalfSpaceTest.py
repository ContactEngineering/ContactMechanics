#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   FFTElasticHalfSpaceTest.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests the fft elastic halfspace implementation

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
    from .PyCoTest import PyCoTestCase
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class PeriodicFFTElasticHalfSpaceTest(PyCoTestCase):
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

    def test_same_energy(self):
        """
        Asserts that the energies computed in the real space and in the fourier space are the same
        """
        for res in [(16,16),(16,15),(15,16),(15,9)]:
            with self.subTest(res=res):
                disp = np.random.normal(size=res)
                hs = PeriodicFFTElasticHalfSpace(res, self.young, self.size)
                np.testing.assert_allclose(hs.evaluate(disp, pot=True, forces=True)[0], hs.evaluate(disp, pot=True, forces=False)[0])

    def test_sineWave_disp(self):
        """
        Compares the computed forces and energies to the analytical solution for single wavevector displacements (cos(kx) + sin(kx))

        This allows to localize problems in the computation of the energy in the k_space

        Parameters
        ----------
        nx
        ny

        Returns
        -------

        """
        for res in [(64,32),(65,32),(64,33),(65,33)]:
            nx, ny = res
            with self.subTest(nx=nx,ny=ny):

                sx = 2.45  # 30.0
                sy = 1.0

                # equivalent Young's modulus
                E_s = 1.0

                for k in [(1, 0), (0, 1), (1, 2), (nx // 2, 0), (1, ny // 2), (0, 2), (nx // 2, ny // 2), (0, ny // 2)]:
                    # print("testing wavevector ({}* np.pi * 2 / sx, {}* np.pi * 2 / sy) ".format(*k))
                    qx = k[0] * np.pi * 2 / sx
                    qy = k[1] * np.pi * 2 / sy
                    q = np.sqrt(qx ** 2 + qy ** 2)

                    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])
                    disp = np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)

                    refpressure = - disp * E_s / 2 * q
                    # np.testing.assert_allclose(refpressure,
                    #    PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                    #    fftengine=NumpyFFTEngine((nx,ny))).evaluate_force(disp) / (sx*sy / (nx*ny)))

                    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),)

                    kpressure = substrate.evaluate_k_force(disp) / substrate.area_per_pt / (nx * ny)
                    expected_k_disp = np.zeros((nx, ny // 2 + 1), dtype=complex)
                    expected_k_disp[k[0], k[1]] += .5 - .5j

                    # add the symetrics
                    if k[1] == 0:
                        expected_k_disp[-k[0], 0] += .5 + .5j
                    if k[1] == ny // 2 and ny % 2 == 0:
                        expected_k_disp[-k[0], k[1]] += .5 + .5j

                    np.testing.assert_allclose(rfftn(disp) / (nx * ny),
                                               expected_k_disp, rtol=1e-7, atol=1e-10)

                    expected_k_pressure = - E_s / 2 * q * expected_k_disp
                    np.testing.assert_allclose(kpressure, expected_k_pressure, rtol=1e-7, atol=1e-10)

                    computedpressure = substrate.evaluate_force(disp) / substrate.area_per_pt
                    np.testing.assert_allclose(computedpressure, refpressure, atol=1e-10, rtol=1e-7)

                    computedenergy_kspace = substrate.evaluate(disp, pot=True, forces=False)[0]
                    computedenergy = substrate.evaluate(disp, pot=True, forces=True)[0]
                    refenergy = E_s / 8 * 2 * q * sx * sy

                    # print(substrate.domain_resolution[-1] % 2)
                    # print(substrate.fourier_resolution)
                    # print(substrate.fourier_location[-1] + substrate.fourier_resolution[-1] - 1)
                    # print(substrate.domain_resolution[-1] // 2 )
                    # print(computedenergy)
                    # print(computedenergy_kspace)
                    # print(refenergy)
                    np.testing.assert_allclose(computedenergy, refenergy, rtol=1e-10,
                                               err_msg="wavevektor {} for domain_resolution {}".format(
                                                   k, substrate.computational_resolution))
                    np.testing.assert_allclose(computedenergy_kspace, refenergy, rtol=1e-10,
                                               err_msg="wavevektor {} for domain_resolution {}".format(
                                                   k, substrate.computational_resolution))

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

    def test_limit_of_large_thickness(self):
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size,
                                         poisson=self.poisson)
        hsf = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size,
                                          poisson=self.poisson, thickness=20)
        diff = hs.weights-hsf.weights
        self.assertArrayAlmostEqual(hs.weights.ravel()[1:],
                                    hsf.weights.ravel()[1:], tol=1e-6)

    def test_no_nans(self):
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young, self.size,
                                         poisson=self.poisson, thickness=100)
        self.assertTrue(np.count_nonzero(np.isnan(hs.weights)) == 0)


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

