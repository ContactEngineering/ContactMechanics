#
# Copyright 2020 Lars Pastewka
#           2020 Antoine Sanner
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
Tests for PyCo helper tools
"""

import unittest

import numpy as np

from ContactMechanics.Tools.ContactAreaAnalysis import distance_map

import pytest
from NuMPI import MPI

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="test only serial functionalities, please execute with pytest")


class ToolTest(unittest.TestCase):
    def test_distance_map(self):
        cmap = np.zeros((10, 10), dtype=bool)
        ind1 = np.random.randint(0, 10)
        ind2 = np.random.randint(0, 10)
        cmap[ind1, ind2] = True
        dmap = distance_map(cmap)
        self.assertAlmostEqual(np.max(dmap), 10 / np.sqrt(2))

        dx = np.abs(dmap - np.roll(dmap, 1))
        dy = np.abs(dmap - np.roll(dmap, 1, axis=1))

        self.assertLessEqual(np.max(dx), np.sqrt(2) + 1e-15)
        self.assertLessEqual(np.max(dy), np.sqrt(2) + 1e-15)
