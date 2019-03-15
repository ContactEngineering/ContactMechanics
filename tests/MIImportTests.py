#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   SurfaceTests.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Tests surface classes

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

#try:
import unittest

from PyCo.Topography.Readers.MIReader import TopographyLoaderMI


# except ImportError as err:
#    import sys
#    print(err)
#    sys.exit(-1)


class MISurfaceTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_topography(self):
        file_path = 'tests/file_format_examples/mi1.mi'

        loader = TopographyLoaderMI(file_path)
        topography = loader.topography()

        self.assertAlmostEqual(topography.size[0], 2.0000000000000002e-005, places=8)
        self.assertAlmostEqual(topography.size[1], 2.0000000000000002e-005, places=8)

        self.assertEqual(topography.resolution[0], 256)
        self.assertEqual(topography.resolution[1], 256)

        self.assertAlmostEqual(topography._heights[0, 0], -4.986900329589844e-07, places=12)