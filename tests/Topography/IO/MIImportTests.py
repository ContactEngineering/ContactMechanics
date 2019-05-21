#
# Copyright 2019 Antoine Sanner
#           2019 Kai Haase
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

import unittest
import os

from PyCo.Topography.IO.MI import MIReader

DATADIR = os.path.join(
    os.path.dirname(
    os.path.dirname(
    os.path.dirname(
    os.path.realpath(__file__)))),
    'file_format_examples')


class MISurfaceTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_read_header(self):
        file_path = os.path.join(DATADIR, 'mi1.mi')

        loader = MIReader(file_path)

        # Check if metadata has been read in correctly
        self.assertEqual(loader.channels()[0],
                         {'DisplayOffset': '8.8577270507812517e-004',
                          'DisplayRange': '1.3109436035156252e-002',
                          'acqMode': 'Main',
                          'label': 'Topography',
                          'range': '2.9025000000000003e+000',
                          'unit': 'um',
                          'direction': 'Trace',
                          'filter': '3rd_order',
                          'name': 'Topography',
                          'trace': 'Trace'})

        self.assertEqual(loader._default_channel, 0)
        self.assertEqual(loader.resolution, (256, 256))

        self.assertAlmostEqual(loader.size[0], 2.0000000000000002e-005, places=8)
        self.assertAlmostEqual(loader.size[1], 2.0000000000000002e-005, places=8)

        # Some metadata value
        self.assertEqual(loader.info()['biasSample'], 'TRUE')

    def test_topography(self):
        file_path = os.path.join(DATADIR, 'mi1.mi')

        loader = MIReader(file_path)

        topography = loader.topography()

        # Check one height value
        self.assertAlmostEqual(topography._heights[0, 0], -0.4986900329589844, places=9)

        # Check out if metadata from global and the channel are both in the result
        # From channel metadata
        self.assertTrue('direction' in topography.info.keys())
        # From global metadata
        self.assertTrue('zDacRange' in topography.info.keys())

        # Check the value of one of the metadata
        self.assertEqual(topography.info['unit'], 'um')
