#
# Copyright 2019 k.o.haase@googlemail.com
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

from PyCo.Topography.Readers.OPDxReader import find_2d_data, read_with_check, read_float, read_double, read_int16, read_int32, read_int64, read_varlen, read_structured, read_name, DektakQuantUnit, read_dimension2d_content, read_quantunit_content, read_named_struct, read_item, TopographyLoaderOPDx


# except ImportError as err:
#    import sys
#    print(err)
#    sys.exit(-1)


class OPDxSurfaceTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_read(self):
        file_path = 'tests/file_format_examples/opdx2.OPDx'

        loader = TopographyLoaderOPDx(file_path)

        self.assertEqual(len(loader.buffer), 9835211)

        # Check if the loader has found the data matrices, should be two
        matrix_start = loader.hash_table["/2D_Data/Image/Matrix"].data.matrix.buf.p
        matrix_length = loader.hash_table["/2D_Data/Image/Matrix"].data.matrix.buf.length
        self.assertEqual(matrix_start, 800)
        self.assertEqual(matrix_length, 4915200)  # 1280 * 960 * 4byte

        matrix_start = loader.hash_table["/2D_Data/Raw/Matrix"].data.matrix.buf.p
        matrix_length = loader.hash_table["/2D_Data/Raw/Matrix"].data.matrix.buf.length
        self.assertEqual(matrix_start, 4916784)
        self.assertEqual(matrix_length, 4915200)

        rawdata, metadata = find_2d_data(loader.hash_table, loader.buffer)["Image"]

        self.assertEqual(rawdata.shape, (960, 1280))

        self.assertAlmostEqual(metadata["Image::Height value"], 35.85522, places=4)
        self.assertEqual(metadata["Image::Height unit"], "Âµm")

        self.assertAlmostEqual(metadata["Image::Width value"], 47.81942, places=4)
        self.assertEqual(metadata["Image::Width unit"], "Âµm")

    def test_topography(self):
        file_path = 'tests/file_format_examples/opdx2.OPDx'

        loader = TopographyLoaderOPDx(file_path)
        topography = loader.topography()

        self.assertAlmostEqual(topography.size[0], 0.00047819, places=8)
        self.assertAlmostEqual(topography.size[1], 0.00035855, places=8)

        self.assertEqual(topography.resolution[0], 960)
        self.assertEqual(topography.resolution[1], 1280)

        self.assertAlmostEqual(topography._heights[0, 0], -7.7315344e-05, places=12)

    def test_read_with_check(self):
        buffer = ['V', 'C', 'A', ' ', 'D', 'A', 'T', 'A', '\x01', '\x00', '\x00', 'U', '\x07', '\x00', '\x00', '\x00']

        pos = 2
        nbytes = 4
        out, pos = read_with_check(buffer, pos, nbytes)

        self.assertEqual(out, ['A', ' ', 'D', 'A'])
        self.assertEqual(pos, 6)

        nbytes = 1
        out, pos = read_with_check(buffer, pos, nbytes)
        self.assertEqual(out, 'T')
        self.assertEqual(pos, 7)

    def test_read_float(self):
        buffer = ['\x12', '\x11', '\x05', '\00']
        pos = 0
        out, pos = read_float(buffer, pos)
        self.assertAlmostEqual(out, 4.65301e-40, places=10)
        self.assertEqual(pos, len(buffer))

    def test_read_double(self):
        buffer = ['\x12', '\x11', '\x05', '\00', '\x12', '\x11', '\x05', '\00']
        pos = 0
        out, pos = read_double(buffer, pos)
        self.assertAlmostEqual(out, 7.04608e-309, places=10)
        self.assertEqual(pos, len(buffer))

    def test_read_int16(self):
        buffer = ['\x12', '\x11']
        pos = 0
        out, pos = read_int16(buffer, pos)
        self.assertEqual(out, 4370)
        self.assertEqual(pos, len(buffer))

    def test_read_int32(self):
        buffer = ['\x12', '\x11', '\xab', '\x4a']
        pos = 0
        out, pos = read_int32(buffer, pos)
        self.assertEqual(out, 1252725010)
        self.assertEqual(pos, len(buffer))

    def test_read_int64(self):
        buffer = ['\x12', '\x11', '\xab', '\x4a', '\xc1', '\x31', '\x95', '\x00']
        pos = 0
        out, pos = read_int64(buffer, pos)
        self.assertEqual(out, 41994477781061906)
        self.assertEqual(pos, len(buffer))

    def test_read_varlen(self):
        buffer = ['\x01', '\xab']
        pos = 0
        out, pos = read_varlen(buffer, pos)
        self.assertEqual(out, 171)
        self.assertEqual(pos, len(buffer))

        buffer = ['\x02', '\x12', '\x11']
        pos = 0
        out, pos = read_varlen(buffer, pos)
        self.assertEqual(out, 4370)
        self.assertEqual(pos, len(buffer))

        buffer = ['\x04', '\x12', '\x11', '\xab', '\x4a']
        pos = 0
        out, pos = read_varlen(buffer, pos)
        self.assertEqual(out, 1252725010)
        self.assertEqual(pos, len(buffer))

    def test_read_structured(self):
        buffer = ['\x01', '\x04', '\x12', '\xca', '\x50', '\x71']
        pos = 0
        out, start, pos = read_structured(buffer, pos)
        self.assertEqual(out, ['\x12', '\xca', '\x50', '\x71'])
        self.assertEqual(start, 2)
        self.assertEqual(pos, len(buffer))

    def test_read_name(self):
        buffer = ['\x02', '\x00', '\x00', '\x00', 'O', 'K']
        pos = 0
        out, pos = read_name(buffer, pos)
        self.assertEqual(out, 'OK')
        self.assertEqual(pos, len(buffer))

    def test_read_dimension2d_content(self):
        buffer = ['\x12', '\x11', '\x05', '\00', '\x12', '\x11', '\x05', '\00',
                  '\x02', '\x00', '\x00', '\x00', 'O', 'K',
                  '\x03', '\x00', '\x00', '\x00', 'A', 'B', 'C',
                  '\x5c', '\x8f', '\xc2', '\xf5', '\x28', '\x5c', '\xe7', '\x3f'] \
                 + ['\x00' for _ in range(12)]  # The extra tail
        pos = 0

        unit = DektakQuantUnit()
        unit, divisor, pos = read_dimension2d_content(buffer, pos, unit)

        self.assertAlmostEqual(unit.value, 7.04608e-309, places=10)
        self.assertEqual(unit.name, 'OK')
        self.assertEqual(unit.symbol, 'ABC')
        self.assertAlmostEqual(divisor, 0.73, places=10)
        self.assertEqual(unit.extra, ['\x00' for _ in range(12)])
        self.assertEqual(pos, len(buffer))

    def test_read_quantunit_content(self):
        buffer = ['\x04', '\x00', '\x00', '\x00', 'N', 'A', 'M', 'E',
                  '\x03', '\x00', '\x00', '\x00', 'S', 'Y', 'M',
                  '\x5c', '\x8f', '\xc2', '\xf5', '\x28', '\x5c', '\xe7', '\x3f'] \
                 + ['\x00' for _ in range(12)]  # The extra tail

        pos = 0

        unit, pos = read_quantunit_content(buffer, pos, is_unit=True)
        self.assertEqual(unit.name, 'NAME')
        self.assertEqual(unit.symbol, 'SYM')
        self.assertAlmostEqual(unit.value, 0.73, places=10)
        self.assertEqual(unit.extra, ['\x00' for _ in range(12)])
        self.assertEqual(pos, len(buffer))

        buffer = ['\x5c', '\x8f', '\xc2', '\xf5', '\x28', '\x5c', '\xe7', '\x3f',
                  '\x04', '\x00', '\x00', '\x00', 'N', 'A', 'M', 'E',
                  '\x03', '\x00', '\x00', '\x00', 'S', 'Y', 'M']

        pos = 0

        unit, pos = read_quantunit_content(buffer, pos, is_unit=False)
        self.assertEqual(unit.name, 'NAME')
        self.assertEqual(unit.symbol, 'SYM')
        self.assertAlmostEqual(unit.value, 0.73, places=10)
        self.assertEqual(unit.extra, [])
        self.assertEqual(pos, len(buffer))

    def test_read_named_struct(self):
        buffer = ['\x04', '\x00', '\x00', '\x00', 'N', 'A', 'M', 'E',
                  '\x01', '\x04', '\x12', '\xca', '\x50', '\x71']
        pos = 0
        typename, out, start, pos = read_named_struct(buffer, pos)
        self.assertEqual(typename, 'NAME')
        self.assertEqual(out, ['\x12', '\xca', '\x50', '\x71'])
        self.assertEqual(start, 10)
        self.assertEqual(pos, len(buffer))

    def test_read_item(self):
        # Test for boolean items ---------------------------------------------------------------------------------------
        buffer = ['\x04', '\x00', '\x00', '\x00', 'B', 'O', 'O', 'L',
                  '\x01',  # ID for DEKTAK_BOOLEAN
                  '\x00']

        pos = 0
        hash_table = dict()
        path = ""

        buffer, pos, hash_table, path = read_item(buffer, pos, hash_table, path)

        self.assertEqual(path, '')
        self.assertEqual(pos, len(buffer))

        item = hash_table['/BOOL']

        self.assertEqual(item.data.b, False)

        # Test for int items -------------------------------------------------------------------------------------------
        buffer = ['\x03', '\x00', '\x00', '\x00', 'I', 'N', 'T',
                  '\x06',  # ID for DEKTAK_SINT32
                  '\x12', '\x11', '\xab', '\x4a']

        pos = 0
        hash_table = dict()
        path = ""

        buffer, pos, hash_table, path = read_item(buffer, pos, hash_table, path)

        self.assertEqual(path, '')
        self.assertEqual(pos, len(buffer))

        item = hash_table['/INT']

        self.assertEqual(item.data.si, 1252725010)

        # Test for Quantity items --------------------------------------------------------------------------------------
        buffer = ['\x09', '\x00', '\x00', '\x00', 'T', 'E', 'S', 'T', '_', 'D', 'A', 'T', 'A',
                  '\x13',  # ID for DEKTAK_QUANTITY
                  '\x01',  # length of length: 1 byte
                  '\x17',  # length: 23 byte
                  '\x5c', '\x8f', '\xc2', '\xf5', '\x28', '\x5c', '\xe7', '\x3f',
                  '\x04', '\x00', '\x00', '\x00', 'N', 'A', 'M', 'E',
                  '\x03', '\x00', '\x00', '\x00', 'S', 'Y', 'M']

        pos = 0
        hash_table = dict()
        path = ""

        buffer, pos, hash_table, path = read_item(buffer, pos, hash_table, path)

        self.assertEqual(path, '')
        self.assertEqual(pos, len(buffer))

        item = hash_table['/TEST_DATA']

        self.assertEqual(item.data.qun.name, 'NAME')
        self.assertEqual(item.data.qun.symbol, 'SYM')
        self.assertAlmostEqual(item.data.qun.value, 0.73, places=10)
        self.assertEqual(item.data.qun.extra, [])









