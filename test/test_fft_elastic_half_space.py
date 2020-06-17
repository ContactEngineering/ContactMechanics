#
# Warning: Could no find author name for junge@cmsjunge
# Copyright 2016-2017, 2019-2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
#           2015-2016 Till Junge
#           2015 junge@cmsjunge
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
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
Tests the fft elastic halfspace implementation
"""

import unittest

import pytest

import numpy as np
from numpy.linalg import norm
from numpy.random import rand, random
from numpy.fft import rfftn, irfftn

from NuMPI import MPI

from ContactMechanics import PeriodicFFTElasticHalfSpace
from ContactMechanics import FreeFFTElasticHalfSpace
import ContactMechanics.Tools as Tools

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="test only serial funcionalities, please execute with pytest")


class PeriodicFFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.physical_sizes = (7.5 + 5 * rand(), 7.5 + 5 * rand())
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3 + 2 * random()
        self.poisson = 0.23

    def test_consistency(self):
        pressure = list()
        base_res = 128
        tol = 1e-4
        for i in (1, 2):
            s_res = base_res * i
            test_res = (s_res, s_res)
            hs = PeriodicFFTElasticHalfSpace(test_res, self.young,
                                             self.physical_sizes)
            forces = np.zeros(test_res)
            forces[:s_res // 2, :s_res // 2] = 1.

            pressure.append(
                hs.evaluate_disp(forces)[::i, ::i] * hs.area_per_pt)
        error = ((pressure[0] - pressure[1]) ** 2).sum().sum() / base_res ** 2
        self.assertTrue(error < tol, "error = {}".format(error))

    def test_parabolic_shape_force(self):
        """ test whether the Elastic energy is a quadratic function of the
            applied force"""
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes)
        force = random(self.res)
        force -= force.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            disp = hs.evaluate_disp(i * force)
            El[i] = hs.evaluate_elastic_energy(i * force, disp)
        tol = 1e-10
        error = norm(El / El[1] - np.arange(nb_tests) ** 2)
        self.assertTrue(error < tol)

    def test_parabolic_shape_disp(self):
        """ test whether the Elastic energy is a quadratic function of the
            applied displacement"""
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes)
        disp = random(self.res)
        disp -= disp.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            force = hs.evaluate_force(i * disp)
            El[i] = hs.evaluate_elastic_energy(i * force, disp)
        tol = 1e-10
        error = norm(El / El[1] - np.arange(nb_tests) ** 2)
        self.assertTrue(error < tol)

    def test_gradient(self):
        res = size = (2, 2)
        disp = random(res)
        disp -= disp.mean()
        hs = PeriodicFFTElasticHalfSpace(res, self.young, size)
        hs.compute(disp, forces=True)
        f = hs.energy
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
        # since only the zero-frequency is rejected, any force/disp field with
        # zero mean should be fully reversible
        tol = 1e-10
        for res in ((self.res[0],), self.res, (self.res[0] + 1, self.res[1]),
                    (self.res[0], self.res[1] + 1)):
            hs = PeriodicFFTElasticHalfSpace(res, self.young,
                                             self.physical_sizes)
            disp = random(res)
            disp -= disp.mean()

            error = Tools.mean_err(disp,
                                   hs.evaluate_disp(hs.evaluate_force(disp)))
            self.assertTrue(
                error < tol,
                "for nb_grid_pts = {}, error = {} > tol = {}"
                .format(res, error, tol))

            force = random(res)
            force -= force.mean()

            error = Tools.mean_err(force,
                                   hs.evaluate_force(hs.evaluate_disp(force)))
            self.assertTrue(
                error < tol,
                "for nb_grid_pts = {}, error = {} > tol = {}".format(
                    res, error, tol))

    # @unittest.skip("wait on 1d support")
    def test_energy(self):
        tol = 1e-10
        L = 2 + rand()  # domain length
        a = 3 + rand()  # amplitude of force
        E = 4 + rand()  # Young's Mod
        for res in [4, 8, 16]:
            area_per_pt = L / res
            x = np.arange(res) * L / res
            force = a * np.cos(2 * np.pi / L * x)

            # theoretical FFT of force
            Fforce = np.zeros_like(x)
            Fforce[1] = Fforce[-1] = res / 2. * a

            # theoretical FFT of disp
            Fdisp = np.zeros_like(x)
            Fdisp[1] = Fdisp[-1] = res / 2. * a / E * L / (2 * np.pi)

            # verify consistency
            hs = PeriodicFFTElasticHalfSpace(res, E, L)
            fforce = rfftn(force.T).T
            fdisp = hs.weights * fforce
            self.assertTrue(
                Tools.mean_err(fforce, Fforce, rfft=True) < tol,
                "fforce = \n{},\nFforce = \n{}".format(
                    fforce.real, Fforce))
            self.assertTrue(
                Tools.mean_err(fdisp, Fdisp, rfft=True) < tol,
                "fdisp = \n{},\nFdisp = \n{}".format(
                    fdisp.real, Fdisp))

            # Fourier energy
            E = .5 * np.dot(Fforce / area_per_pt, Fdisp) / res

            disp = hs.evaluate_disp(force)
            e = hs.evaluate_elastic_energy(force, disp)
            kdisp = hs.evaluate_k_disp(force)
            ee = hs.evaluate_elastic_energy_k_space(fforce, kdisp)
            self.assertTrue(
                abs(e - ee) < tol,
                "violate Parseval: e = {}, ee = {}, ee/e = {}".format(
                    e, ee, ee / e))

            self.assertTrue(
                abs(E - e) < tol,
                "theoretical E = {}, computed e = {}, "
                "diff(tol) = {}({})".format(E, e, E - e, tol))

    def test_same_energy(self):
        """
        Asserts that the energies computed in the real space and in the fourier
        space are the same
        """
        for res in [(16, 16), (16, 15), (15, 16), (15, 9)]:
            disp = np.random.normal(size=res)
            hs = PeriodicFFTElasticHalfSpace(res, self.young,
                                             self.physical_sizes)
            np.testing.assert_allclose(
                hs.evaluate(disp, pot=True, forces=True)[0],
                hs.evaluate(disp, pot=True, forces=False)[0])

    def test_unit_neutrality(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c = 1. + np.random.rand()
        force_c = 1. + np.random.rand()
        pressure_c = force_c / length_c ** 2
        # energy_c = force_c * length_c

        # length_rc = (1., 1. / length_c)
        force_rc = (1., 1. / force_c)
        # pressure_rc = (1., 1. / pressure_c)
        # energy_rc = (1., 1. / energy_c)
        nb_grid_pts = (32, 32)
        young = (self.young, pressure_c * self.young)
        size = self.physical_sizes[0], 2 * self.physical_sizes[1]
        size = (size, tuple((length_c * s for s in size)))
        # print('SELF.SIZE = {}'.format(self.physical_sizes))

        disp = np.random.random(nb_grid_pts)
        disp -= disp.mean()
        disp = (disp, disp * length_c)

        forces = list()
        for i in range(2):
            sub = PeriodicFFTElasticHalfSpace(nb_grid_pts, young[i], size[i])
            force = sub.evaluate_force(disp[i])
            forces.append(force * force_rc[i])
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

    def test_unit_neutrality1D(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c = 1. + np.random.rand()
        force_c = 2. + np.random.rand()
        pressure_c = force_c / length_c ** 2
        # energy_c = force_c * length_c
        force_per_length_c = force_c / length_c

        # length_rc = (1., 1. / length_c)
        # force_rc = (1., 1. / force_c)
        # pressure_rc = (1., 1. / pressure_c)
        # energy_rc = (1., 1. / energy_c)
        force_per_length_rc = (1., 1. / force_per_length_c)

        nb_grid_pts = (32,)
        young = (self.young, pressure_c * self.young)
        size = (self.physical_sizes[0], length_c * self.physical_sizes[0])

        disp = np.random.random(nb_grid_pts)
        disp -= disp.mean()
        disp = (disp, disp * length_c)

        subs = tuple(
            (PeriodicFFTElasticHalfSpace(nb_grid_pts, y, s) for y, s in
             zip(young, size)))
        forces = tuple((s.evaluate_force(d) * f_p_l for s, d, f_p_l in
                        zip(subs, disp, force_per_length_rc)))
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))

    def test_uniform_displacement(self):
        """ test whether uniform displacement returns stiffness_q0"""
        sq0 = 1.43
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes, stiffness_q0=sq0)
        force = hs.evaluate_force(-np.ones(self.res))
        self.assertAlmostEqual(force.sum() / np.prod(self.physical_sizes), sq0)

    def test_uniform_displacement_finite_height(self):
        """ test whether uniform displacement returns stiffness_q0"""
        h0 = 3.45
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes, thickness=h0,
                                         poisson=self.poisson)
        force = hs.evaluate_force(-np.ones(self.res))
        M = (1 - self.poisson) / (
                    (1 - 2 * self.poisson) * (1 + self.poisson)) * self.young
        self.assertAlmostEqual(force.sum() / np.prod(self.physical_sizes),
                               M / h0)

    def test_limit_of_large_thickness(self):
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes,
                                         poisson=self.poisson)
        hsf = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                          self.physical_sizes, thickness=20,
                                          poisson=self.poisson)
        # diff = hs.weights - hsf.weights
        np.testing.assert_allclose(hs.weights.ravel()[1:],
                                   hsf.weights.ravel()[1:], atol=1e-6)

    def test_no_nans(self):
        hs = PeriodicFFTElasticHalfSpace(self.res, self.young,
                                         self.physical_sizes, thickness=100,
                                         poisson=self.poisson)
        self.assertTrue(np.count_nonzero(np.isnan(hs.weights)) == 0)

    # TODO: Test independence of result of x and y Direction,
    # this is already in the MPI variant


class FreeFFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.physical_sizes = (7.5 + 5 * rand(), 7.5 + 5 * rand())
        base_res = 16
        self.res = (base_res, base_res)
        self.young = 3 + 2 * random()

    def test_consistency(self):
        pressure = list()
        base_res = 32
        tol = 1e-1
        for i in (1, 2):
            s_res = base_res * i
            test_res = (s_res, s_res)
            hs = FreeFFTElasticHalfSpace(test_res, self.young,
                                         self.physical_sizes)
            forces = np.zeros([2 * r for r in test_res])
            forces[:s_res // 2, :s_res // 2] = 1.

            pressure.append(
                hs.evaluate_disp(forces)[::i, ::i] * hs.area_per_pt)
        error = ((pressure[0] - pressure[1]) ** 2).sum().sum() / base_res ** 2
        self.assertTrue(error < tol, "error = {}, tol = {}".format(error, tol))

    def test_rfftn(self):
        force = np.zeros([2 * r for r in self.res])

        force[:self.res[0], :self.res[1]] = np.random.random(self.res)
        from muFFT import FFT
        ref = rfftn(force.T).T
        fftengine = FFT([2 * r for r in self.res], fft="serial")
        fftengine.initialise(1)
        tested = np.zeros(fftengine.nb_fourier_grid_pts, order='f',
                          dtype=complex)
        fftengine.fft(force, tested)
        np.testing.assert_allclose(ref.real,
                                   tested.real)
        np.testing.assert_allclose(ref.imag,
                                   tested.imag)

    def test_fftengine_nb_grid_pts(self):
        hs = FreeFFTElasticHalfSpace(self.res, self.young, self.physical_sizes)
        assert hs.fftengine.nb_domain_grid_pts == tuple(
            [2 * r for r in self.res])

    def test_temp(self):  # TODO: Remove me
        hs = FreeFFTElasticHalfSpace(self.res, self.young, self.physical_sizes)
        force = np.zeros(hs.nb_domain_grid_pts)
        assert hs.nb_domain_grid_pts == tuple([2 * r for r in self.res])
        force[:self.res[0], :self.res[1]] = np.random.random(self.res)
        force[:self.res[0], :self.res[1]] -= \
            force[:self.res[0], :self.res[1]].mean()

        # np.testing.assert_allclose(rfftn(-force), hs.fftengine.fft(-force))
        # hs.fftengine.fft(-force)
        kdisp_hs = hs.weights * rfftn(-force.T).T / hs.area_per_pt
        kdisp = hs.evaluate_k_disp(force)

        np.testing.assert_allclose(kdisp_hs, kdisp, rtol=1e-10)
        np.testing.assert_allclose(kdisp_hs.real, kdisp.real)
        np.testing.assert_allclose(kdisp_hs.imag, kdisp.imag)

        kforce = rfftn(force.T).T
        np_pts = np.prod(hs.nb_domain_grid_pts)
        # area_per_pt = np.prod(self.physical_sizes) / np_pts
        energy = .5 * (np.vdot(-kforce, kdisp) +
                       np.vdot(-kforce[1:-1, ...], kdisp[1:-1, ...])) / np_pts
        error = abs(energy.imag)
        tol = 1e-10
        self.assertTrue(error < tol,
                        "error (imaginary part) = {}, tol = {}".format(
                            error, tol))
        error = abs(energy - hs.evaluate_elastic_energy_k_space(kforce, kdisp))
        self.assertTrue(error < tol,
                        ("error (comparison) = {}, tol = {}, energy = {}, "
                         "kenergy = {}").format(
                            error, tol, energy,
                            hs.evaluate_elastic_energy_k_space(kforce, kdisp)))

    def test_realnessEnergy(self):
        hs = FreeFFTElasticHalfSpace(self.res, self.young, self.physical_sizes)
        force = np.zeros(hs.nb_domain_grid_pts)
        force[:self.res[0], :self.res[1]] = np.random.random(self.res)
        force[:self.res[0], :self.res[1]] -= \
            force[:self.res[0], :self.res[1]].mean()
        kdisp = hs.evaluate_k_disp(force)
        kforce = rfftn(force.T).T
        np_pts = np.prod(hs.nb_domain_grid_pts)
        energy = .5 * (np.vdot(-kforce, kdisp) +
                       np.vdot(-kforce[1:-1, ...], kdisp[1:-1, ...])) / np_pts
        error = abs(energy.imag)
        tol = 1e-10
        self.assertTrue(error < tol,
                        "error (imaginary part) = {}, tol = {}".format(
                            error, tol))
        error = abs(energy - hs.evaluate_elastic_energy_k_space(kforce, kdisp))
        self.assertTrue(error < tol,
                        ("error (comparison) = {}, tol = {}, energy = {}, "
                         "kenergy = {}").format(
                            error, tol, energy,
                            hs.evaluate_elastic_energy_k_space(kforce, kdisp)))

    def test_energy(self):
        tol = 1e-10
        L = 2 + rand()  # domain length
        a = 3 + rand()  # amplitude of force
        E = 4 + rand()  # Young's Mod
        for res in [4, 8, 16]:
            area_per_pt = L / res
            x = np.arange(res) * area_per_pt
            force = a * np.cos(2 * np.pi / L * x)

            # theoretical FFT of force
            Fforce = np.zeros_like(x)
            Fforce[1] = Fforce[-1] = res / 2. * a

            # theoretical FFT of disp
            Fdisp = np.zeros_like(x)
            Fdisp[1] = Fdisp[-1] = res / 2. * a / E * L / (2. * np.pi)

            # verify consistency
            hs = PeriodicFFTElasticHalfSpace(res, E, L)
            fforce = rfftn(force.T).T
            fdisp = hs.weights * fforce
            self.assertTrue(
                Tools.mean_err(fforce, Fforce, rfft=True) < tol,
                "fforce = \n{},\nFforce = \n{}".format(
                    fforce.real, Fforce))
            self.assertTrue(
                Tools.mean_err(fdisp, Fdisp, rfft=True) < tol,
                "fdisp = \n{},\nFdisp = \n{}".format(
                    fdisp.real, Fdisp))

            # Fourier energy
            E = .5 * np.dot(Fforce / area_per_pt, Fdisp) / res

            disp = hs.evaluate_disp(force)
            e = hs.evaluate_elastic_energy(force, disp)
            kdisp = hs.evaluate_k_disp(force)
            self.assertTrue(abs(disp - irfftn(kdisp.T).T).sum() < tol,
                            ("disp   = {}\n"
                             "ikdisp = {}").format(disp, irfftn(kdisp.T).T))
            ee = hs.evaluate_elastic_energy_k_space(fforce, kdisp)
            self.assertTrue(
                abs(e - ee) < tol,
                "violate Parseval: e = {}, ee = {}, ee/e = {}".format(
                    e, ee, ee / e))

            self.assertTrue(
                abs(E - e) < tol,
                "theoretical E = {}, computed e = {}, diff(tol) = {}({})"
                .format(E, e, E - e, tol))

    def test_unit_neutrality(self):
        tol = 1e-7
        # runs the same problem in two unit sets and checks whether results are
        # changed

        # Conversion factors
        length_c = 1. + np.random.rand()
        force_c = 2. + np.random.rand()
        pressure_c = force_c / length_c ** 2
        # energy_c = force_c * length_c

        # length_rc = (1., 1. / length_c)
        force_rc = (1., 1. / force_c)
        # pressure_rc = (1., 1. / pressure_c)
        # energy_rc = (1., 1. / energy_c)
        nb_grid_pts = (32, 32)
        young = (self.young, pressure_c * self.young)
        size = (self.physical_sizes,
                tuple((length_c * s for s in self.physical_sizes)))

        comp_nb_grid_pts = tuple((2 * res for res in nb_grid_pts))
        disp = np.random.random(comp_nb_grid_pts)
        disp -= disp.mean()
        disp = (disp, disp * length_c)

        forces = list()
        for i in range(2):
            sub = FreeFFTElasticHalfSpace(nb_grid_pts, young[i], size[i])
            force = sub.evaluate_force(disp[i])
            forces.append(force * force_rc[i])
        error = Tools.mean_err(forces[0], forces[1])
        self.assertTrue(error < tol,
                        "error = {} ≥ tol = {}".format(error, tol))
