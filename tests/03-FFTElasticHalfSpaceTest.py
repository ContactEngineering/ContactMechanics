#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   03-FFTElasticHalfSpaceTest.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   27 Jan 2015
#
# @brief  Tests the fft elastic halfspace implementation
#
# @section LICENCE
#
#  Copyright (C) 2015 Till Junge
#
# PyPyContact is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# PyPyContact is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

import unittest
import numpy as np
from numpy.linalg import norm
from numpy.random import rand, random
from scipy.fftpack import fftn, ifftn

from PyPyContact.SolidMechanics import FFTElasticHalfSpace

def mean_err(arr1, arr2):
    return abs(np.ravel(arr1-arr2)).mean()

class FFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*rand(), 7.5+5*rand())
        base_res = 128
        self.res = (base_res, base_res)
        self.young = 3+2*random()

    def test_consistency(self):
        pressure = list()
        base_res = 128
        tol = 1e-5
        for i in (1, 2):
            s_res = base_res*i
            test_res = (s_res, s_res)
            hs = FFTElasticHalfSpace(test_res, self.young, self.size)
            forces = np.zeros(test_res)
            forces[:s_res//2,:s_res//2] = 1.

            pressure.append(hs.evaluateDisp(forces)[::i,::i]*hs.area_per_pt)
        error = ((pressure[0]-pressure[1])**2).sum().sum()/base_res**2
        self.assertTrue(error < tol)


    def test_parabolic_shape_force(self):
        """ tests whether the Elastic energy is a quadratic function of the
            applied force"""
        hs = FFTElasticHalfSpace(self.res, self.young, self.size)
        force = random(self.res)
        force -= force.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            disp = hs.evaluateDisp(i*force)
            El[i] = hs.evaluateElasticEnergy(i*force, disp)
        tol = 1e-10
        error = norm(El/El[1]-np.arange(nb_tests)**2)
        self.assertTrue(error<tol)

    def test_parabolic_shape_disp(self):
        """ tests whether the Elastic energy is a quadratic function of the
            applied displacement"""
        hs = FFTElasticHalfSpace(self.res, self.young, self.size)
        disp = random(self.res)
        disp -= disp.mean()
        nb_tests = 4
        El = np.zeros(nb_tests)
        for i in range(nb_tests):
            force = hs.evaluateForce(i*disp)
            El[i] = hs.evaluateElasticEnergy(i*force, disp)
        tol = 1e-10
        error = norm(El/El[1]-np.arange(nb_tests)**2)
        self.assertTrue(error<tol)

    def test_force_disp_reversibility(self):
        ## since only the zero-frequency is rejected, any force/disp field with
        ## zero mean should be fully reversible
        tol = 1e-10
        for res in ((self.res[0],), self.res):
            hs = FFTElasticHalfSpace(res, self.young, self.size)
            disp = random(res)
            disp -= disp.mean()

            error = mean_err(disp, hs.evaluateDisp(hs.evaluateForce(disp)))
            self.assertTrue(
                error < tol,
                "for resolution = {}, error = {} > tol = {}".format(
                    res, error, tol))

            force = random(res)
            force -= force.mean()

            error = mean_err(force, hs.evaluateForce(hs.evaluateDisp(force)))
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
            Fdisp[1] = Fdisp[-1] = res/2.*a/E

            ## verify consistency
            hs = FFTElasticHalfSpace(res, E, l)
            fforce = fftn(force)
            fdisp = hs.weights*fforce
            self.assertTrue(
                mean_err(fforce, Fforce)<tol, "fforce = \n{},\nFforce = \n{}".format(
                    fforce.real, Fforce))
            self.assertTrue(
                mean_err(fdisp, Fdisp)<tol, "fdisp = \n{},\nFdisp = \n{}".format(
                    fdisp.real, Fdisp))

            ##Fourier energy
            E = .5*np.dot(Fforce/area_per_pt, Fdisp)/res

            disp = hs.evaluateDisp(force)
            e =  hs.evaluateElasticEnergy(force, disp)
            kdisp = hs.evaluateKDisp(force)
            ee = hs.evaluateElasticEnergyKspace(fforce, kdisp)
            self.assertTrue(
                abs(e-ee) < tol,
                 "violate Parseval: e = {}, ee = {}, ee/e = {}".format(
                    e, ee, ee/e))

            self.assertTrue(
                abs(E-e)<tol,
                "theoretical E = {}, computed e = {}, diff(tol) = {}({})".format(
                    E, e, E-e, tol))
