#
# Copyright 2019 Lars Pastewka
#           2018 Antoine Sanner
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
Tests tools for analysis of contact geometries.
"""

from math import sqrt
import unittest

import numpy as np

from PyCo.Tools.ContactAreaAnalysis import (assign_patch_numbers, assign_segment_numbers, distance_map,
                                            inner_perimeter, outer_perimeter)
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
