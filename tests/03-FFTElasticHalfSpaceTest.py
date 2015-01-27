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

from PyPyContact.SolidMechanics import FFTElasticHalfSpace

class FFTElasticHalfSpaceTest(unittest.TestCase):
    def setUp(self):
        self.size = (7.5+5*np.random.rand(), 7.5+5*np.random.rand())
        self.young = 3+2*np.random.random()

    def test_consistency(self):
        results = list()
        base_res = 128
        tol = 1e-5
        for i in (1, 2):
            s_res = base_res*i
            test_res = (s_res, s_res)
            hs = FFTElasticHalfSpace(test_res, self.young, self.size)
            forces = np.zeros(test_res)
            forces[:s_res//2,:s_res//2] = 1.

            results.append(hs.evaluate_disp(forces)[::i,::i])
        error = ((results[0]-results[1])**2).sum().sum()/base_res**2
        self.assertTrue(error < tol)
