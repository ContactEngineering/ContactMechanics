#
# Copyright 2017, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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
Tests tools for analysis of contact geometries.
"""

import os
import unittest
from math import sqrt

import numpy as np
import pytest

from NuMPI import MPI

from ContactMechanics.Tools.ContactAreaAnalysis import assign_patch_numbers, \
    assign_segment_numbers, distance_map, inner_perimeter, outer_perimeter, \
    patch_areas

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="test only serial functionalities, please execute with pytest")


class TestAnalysis(unittest.TestCase):

    def test_assign_patch_numbers(self):
        m_xy = np.zeros([3, 3], dtype=bool)
        m_xy[1, 1] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 1)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0]]))

        m_xy = np.zeros([5, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 2)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0],
                                                   [0, 2, 0],
                                                   [0, 0, 0]]))

        m_xy = np.zeros([6, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True
        m_xy[3, 2] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 2)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0],
                                                   [0, 2, 2],
                                                   [0, 0, 0],
                                                   [0, 0, 0]]))

        m_xy = np.zeros([6, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[2, 1] = True
        m_xy[3, 1] = True
        m_xy[3, 2] = True

        nump, p_xy = assign_patch_numbers(m_xy)

        self.assertEqual(nump, 1)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 1, 0],
                                                   [0, 1, 1],
                                                   [0, 0, 0],
                                                   [0, 0, 0]]))

        m_xy = np.loadtxt('{}/contact_map.txt.gz'.format(
            os.path.dirname(os.path.realpath(__file__))), dtype=bool)

        # This is a regression test
        ref_patch_areas = [403, 4, 210, 46, 1, 3, 2, 2, 16, 2977, 2, 11, 1, 13,
                           2, 3, 5, 2, 1, 2, 1, 1, 2, 2, 1, 5, 1,
                           25, 2, 6526, 370, 1, 1, 1, 3, 1, 10, 4, 1, 5, 6, 24,
                           7, 1, 5, 16, 1, 10, 3, 1, 71, 1, 2, 1,
                           1, 1, 1, 5, 4, 1, 2, 1, 6, 2, 5, 190, 17, 2, 2, 2,
                           10, 1, 1, 16, 1, 1, 1, 92, 8, 1, 1, 1, 2,
                           1, 3, 3, 1, 33, 5, 4, 6, 3, 6, 43, 1, 4, 5, 1, 6, 4,
                           1, 1, 2, 15, 1, 3, 1, 1, 2, 2, 1, 28,
                           3, 1, 1, 2, 1, 11, 2, 2, 2, 3, 1]

        nump, p_xy = assign_patch_numbers(m_xy)
        self.assertEqual(nump, 123)
        np.testing.assert_allclose(patch_areas(p_xy), ref_patch_areas)

    def test_assign_segment_numbers(self):
        m_xy = np.zeros([3, 3], dtype=bool)
        m_xy[1, 1] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 1)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0]]))

        m_xy = np.zeros([5, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 2)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0],
                                                   [0, 2, 0],
                                                   [0, 0, 0]]))

        m_xy = np.zeros([6, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[2, 1] = True
        m_xy[3, 1] = True
        m_xy[3, 2] = True

        nump, p_xy = assign_segment_numbers(m_xy)

        self.assertEqual(nump, 3)
        np.testing.assert_allclose(p_xy, np.array([[0, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 2, 0],
                                                   [0, 3, 3],
                                                   [0, 0, 0],
                                                   [0, 0, 0]]))

    def test_distance_map(self):
        m_xy = np.zeros([3, 3], dtype=bool)
        m_xy[1, 1] = True

        d_xy = distance_map(m_xy)

        sqrt_2 = sqrt(2.0)
        np.testing.assert_allclose(d_xy, np.array([[sqrt_2, 1.0, sqrt_2],
                                                   [1.0, 0.0, 1.0],
                                                   [sqrt_2, 1.0, sqrt_2]]))

        m_xy = np.zeros([5, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True

        d_xy = distance_map(m_xy)

        np.testing.assert_allclose(d_xy, np.array([[sqrt_2, 1.0, sqrt_2],
                                                   [1.0, 0.0, 1.0],
                                                   [sqrt_2, 1.0, sqrt_2],
                                                   [1.0, 0.0, 1.0],
                                                   [sqrt_2, 1.0, sqrt_2]]))

        m_xy = np.zeros([6, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True

        d_xy = distance_map(m_xy)

        sqrt_5 = sqrt(5.0)
        np.testing.assert_allclose(d_xy, np.array([[sqrt_2, 1.0, sqrt_2],
                                                   [1.0, 0.0, 1.0],
                                                   [sqrt_2, 1.0, sqrt_2],
                                                   [1.0, 0.0, 1.0],
                                                   [sqrt_2, 1.0, sqrt_2],
                                                   [sqrt_5, 2.0, sqrt_5]]))

    def test_perimeter(self):
        m_xy = np.zeros([3, 3], dtype=bool)
        m_xy[1, 1] = True

        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)

        self.assertTrue(np.array_equal(i_xy, m_xy))
        self.assertTrue(np.array_equal(o_xy, np.array([[False, True, False],
                                                       [True, False, True],
                                                       [False, True, False]])))

        m_xy = np.zeros([5, 3], dtype=bool)
        m_xy[1, 1] = True
        m_xy[3, 1] = True

        i_xy = inner_perimeter(m_xy)
        o_xy = outer_perimeter(m_xy)

        self.assertTrue(np.array_equal(i_xy, np.array([[False, False, False],
                                                       [False, True, False],
                                                       [False, False, False],
                                                       [False, True, False],
                                                       [False, False,
                                                        False]])))
        self.assertTrue(np.array_equal(o_xy, np.array([[False, True, False],
                                                       [True, False, True],
                                                       [False, True, False],
                                                       [True, False, True],
                                                       [False, True, False]])))


###

if __name__ == '__main__':
    unittest.main()
