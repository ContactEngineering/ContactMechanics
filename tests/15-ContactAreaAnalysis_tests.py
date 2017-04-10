# -*- coding:utf-8 -*-
"""
@file   15-ContactAreaAnalysis_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Apr 2017

@brief  Tests tools for analysis of contact geometries.

@section LICENCE

 Copyright (C) 2017 Lars Pastewka

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

from math import sqrt
import unittest

import numpy as np

from PyCo.Tools import assign_patch_numbers, assign_segment_numbers
from PyCo.Tools import distance_map, inner_perimeter, outer_perimeter
from .PyCoTest import PyCoTestCase

###

class TestAnalysis(PyCoTestCase):

    def test_assign_patch_numbers(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 1)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                                    [0,1,0],
                                                    [0,0,0]]))

        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 2)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                                    [0,1,0],
                                                    [0,0,0],
                                                    [0,2,0],
                                                    [0,0,0]]))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 2)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                                    [0,1,0],
                                                    [0,0,0],
                                                    [0,2,2],
                                                    [0,0,0],
                                                    [0,0,0]]))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[2,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 1)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                                    [0,1,0],
                                                    [0,1,0],
                                                    [0,1,1],
                                                    [0,0,0],
                                                    [0,0,0]]))


    def test_assign_segment_numbers(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 1)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                              [0,1,0],
                                              [0,0,0]]))

        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 2)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                              [0,1,0],
                                              [0,0,0],
                                              [0,2,0],
                                              [0,0,0]]))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[2,1] = True
        m_xy[3,1] = True
        m_xy[3,2] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 3)
        self.assertArrayAlmostEqual(p_xy, np.array([[0,0,0],
                                                    [0,1,0],
                                                    [0,2,0],
                                                    [0,3,3],
                                                    [0,0,0],
                                                    [0,0,0]]))


    def test_distance_map(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True

        d_xy = distance_map(m_xy)

        sqrt_2 = sqrt(2.0)
        self.assertArrayAlmostEqual(d_xy, np.array([[sqrt_2,1.0,sqrt_2],
                                                    [1.0,0.0,1.0],
                                                    [sqrt_2,1.0,sqrt_2]]))

        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        d_xy = distance_map(m_xy)

        self.assertArrayAlmostEqual(d_xy, np.array([[sqrt_2,1.0,sqrt_2],
                                                    [1.0,0.0,1.0],
                                                    [sqrt_2,1.0,sqrt_2],
                                                    [1.0,0.0,1.0],
                                                    [sqrt_2,1.0,sqrt_2]]))

        m_xy = np.zeros([6,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        d_xy = distance_map(m_xy)

        sqrt_5 = sqrt(5.0)
        self.assertArrayAlmostEqual(d_xy, np.array([[sqrt_2,1.0,sqrt_2],
                                                    [1.0,0.0,1.0],
                                                    [sqrt_2,1.0,sqrt_2],
                                                    [1.0,0.0,1.0],
                                                    [sqrt_2,1.0,sqrt_2],
                                                    [sqrt_5,2.0,sqrt_5]]))


    def test_perimeter(self):
        m_xy = np.zeros([3,3], dtype=bool)
        m_xy[1,1] = True

        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)

        self.assertTrue(np.array_equal(i_xy, m_xy))
        self.assertTrue(np.array_equal(o_xy, np.array([[False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False]])))

        m_xy = np.zeros([5,3], dtype=bool)
        m_xy[1,1] = True
        m_xy[3,1] = True

        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)

        self.assertTrue(np.array_equal(i_xy, np.array([[False,False,False],
                                                       [False,True, False],
                                                       [False,False,False],
                                                       [False,True, False],
                                                       [False,False,False]])))
        self.assertTrue(np.array_equal(o_xy, np.array([[False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False],
                                                       [True, False,True ],
                                                       [False,True, False]])))

###

if __name__ == '__main__':
    unittest.main()
