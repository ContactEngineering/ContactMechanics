from NuMPI import MPI
import pytest

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial functionalities, please execute with pytest")

import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal

import os
import io
import pickle

from PyCo.Topography import open_topography, read_topography

from PyCo.Topography.IO.FromFile import read_xyz

from PyCo.Topography.IO.FromFile import is_binary_stream
from PyCo.Topography.IO import detect_format

import PyCo.Topography.IO
from PyCo.Topography.IO import readers


###

DATADIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../file_format_examples')

@pytest.mark.parametrize("reader", readers.values())
def test_closes_file_on_failure(reader):
    """
    Tests for each reader class that he doesn't raise a Resourcewarning
    """
    fn = os.path.join(DATADIR, "wrongnpyfile.npy")
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter(
            "always")  # deactivate hiding of ResourceWarnings

        try:
            reader(fn)
        except Exception:
            pass
        # assert no warning is a ResourceWarning
        for wi in w:
            assert not issubclass(wi.category, ResourceWarning)


def test_uniform_stylus():
    t = read_topography(os.path.join(DATADIR, 'example7.txt'))
    assert t.is_uniform


class IOTest(unittest.TestCase):
    def setUp(self):
        self.binary_example_file_list = [
            os.path.join(DATADIR, 'example1.di'),
            os.path.join(DATADIR, 'example.ibw'),
            os.path.join(DATADIR, 'example1.mat'),
            os.path.join(DATADIR, 'example.opd'),
            os.path.join(DATADIR, 'example.x3p'),
            os.path.join(DATADIR, 'example2.x3p'),
        ]
        self.text_example_file_list = [
            os.path.join(DATADIR, 'example.asc'),
            os.path.join(DATADIR, 'example1.txt'),
            os.path.join(DATADIR, 'example2.txt'),
            os.path.join(DATADIR, 'example3.txt'),
            os.path.join(DATADIR, 'example4.txt'),
            os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc'),
        ]
        self.text_example_memory_list = [
            """
            0 0
            1 2
            2 4
            3 6
            """
        ]

    def test_keep_file_open(self):
        for fn in self.text_example_file_list:
            # Text file can be opened as binary or text
            with open(fn, 'rb') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
            with open(fn, 'r') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
        for fn in self.binary_example_file_list:
            with open(fn, 'rb') as f:
                open_topography(f)
                self.assertFalse(f.closed, msg=fn)
        for datastr in self.text_example_memory_list:
            with io.StringIO(datastr) as f:
                open_topography(f)
                self.assertFalse(f.closed, msg="text memory stream for '{}' was closed".format(datastr))

            # Doing the same when but only giving a binary stream
            with io.BytesIO(datastr.encode(encoding='utf-8')) as f:
                open_topography(f)
                self.assertFalse(f.closed, msg="binary memory stream for '{}' was closed".format(datastr))

    def test_is_binary_stream(self):

        # just grep a random existing file here
        fn = self.text_example_file_list[0]

        self.assertTrue(is_binary_stream(open(fn, mode='rb')))
        self.assertFalse(is_binary_stream(open(fn, mode='r')))  # opened as text file

        # should also work with streams in memory
        self.assertTrue(is_binary_stream(io.BytesIO(b"11111")))  # some bytes in memory
        self.assertFalse(is_binary_stream(io.StringIO("11111")))  # some bytes in memory

    def test_can_be_pickled(self):
        file_list = self.text_example_file_list + self.binary_example_file_list

        for fn in file_list:
            print(fn)
            reader = open_topography(fn)
            t = reader.topography(physical_sizes=reader.physical_sizes
            if reader.physical_sizes is not None
            else [1., ] * len(reader.nb_grid_pts))
            s = pickle.dumps(t)
            pickled_t = pickle.loads(s)

            #
            # Compare some attributes after unpickling
            #
            # sometimes the result is a list of topographies
            multiple = isinstance(t, list)
            if not multiple:
                t = [t]
                pickled_t = [pickled_t]

            for x, y in zip(t, pickled_t):
                for attr in ['dim', 'physical_sizes']:
                    assert getattr(x, attr) == getattr(y, attr)
                if x.physical_sizes is not None:
                    assert_array_equal(x.positions(), y.positions())
                    assert_array_equal(x.heights(), y.heights())


class UnknownFileFormatGivenTest(unittest.TestCase):

    def test_read(self):
        with self.assertRaises(PyCo.Topography.IO.UnknownFileFormatGiven):
            PyCo.Topography.IO.open_topography(os.path.join(DATADIR, "surface.2048x2048.h5"),
                                               format='Nonexistentfileformat')

    def test_detect_format(self):
        with self.assertRaises(PyCo.Topography.IO.UnknownFileFormatGiven):
            PyCo.Topography.IO.open_topography(os.path.join(DATADIR, "surface.2048x2048.h5"),
                                               format='Nonexistentfileformat')


class FileFormatMismatchTest(unittest.TestCase):
    def test_read(self):
        with self.assertRaises(PyCo.Topography.IO.FileFormatMismatch):
            PyCo.Topography.IO.open_topography(os.path.join(DATADIR, 'surface.2048x2048.h5'), format="npy")


class LineScanInFileWithMinimalSpacesTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_detect_format_then_read(self):
        self.assertEqual(detect_format(os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc')), 'xyz')

    def test_read(self):
        surface = read_xyz(os.path.join(DATADIR, 'line_scan_1_minimal_spaces.asc'))

        self.assertFalse(surface.is_uniform)
        self.assertEqual(surface.dim, 1)

        x, y = surface.positions_and_heights()
        self.assertGreater(len(x), 0)
        self.assertEqual(len(x), len(y))
