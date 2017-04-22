# -*- coding:utf-8 -*-
"""
@file   __init__.py

@author Lars Pastewka <lars.pastewka@kit.edu>

@date   25 April 2015

@brief  Advanced assertions for Python unittests.
        Taken from matscipy (https://github.com/libAtoms/matscipy)

@section LICENCE

 Copyright (C) 2015 Lars Pastewka

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

from __future__ import print_function

import unittest
import logging
from io import StringIO
import numpy as np

def string_to_array(s):
    return np.loadtxt(StringIO(s)).T

class PyCoTestCase(unittest.TestCase):
    """
    Subclass of unittest.TestCase with extra methods for comparing arrays and dictionaries
    """

    def assertDictionariesEqual(self, d1, d2, skip_keys=[], ignore_case=True):

        def lower_if_ignore_case(k):
            if ignore_case:
                return k.lower()
            else:
                return k

        d1 = dict([(lower_if_ignore_case(k),v) for (k,v) in d1.iteritems() if k not in skip_keys])
        d2 = dict([(lower_if_ignore_case(k),v) for (k,v) in d2.iteritems() if k not in skip_keys])

        if sorted(d1.keys()) != sorted(d2.keys()):
            self.fail('Dictionaries differ: d1.keys() (%r) != d2.keys() (%r)'  % (d1.keys(), d2.keys()))
        for key in d1:
            v1, v2 = d1[key], d2[key]
            if isinstance(v1, np.ndarray):
                try:
                    self.assertArrayAlmostEqual(v1, v2)
                except AssertionError:
                    print(key, v1, v2)
                    raise
            else:
                if v1 != v2:
                    self.fail('Dictionaries differ: key=%s value1=%r value2=%r' % (key, v1, v2))

    def assertEqual(self, a, b):
        if a == b:
            return
        # Repeat comparison with debug-level logging
        import logging
        level = logging.root.level
        logging.root.setLevel(logging.DEBUG)
        a == b
        logging.root.setLevel(level)
        self.fail('%s != %s' % (a,b))


    def assertArrayAlmostEqual(self, a, b, tol=1e-7):
        a = np.array(a)
        b = np.array(b)
        self.assertEqual(a.shape, b.shape)

        if np.isnan(a).any() or np.isnan(b).any():
            print('a')
            print(a)
            print('b')
            print(b)
            self.fail('Not a number (NaN) found in array')

        if a.dtype.kind != 'f':
            self.assertTrue((a == b).all())
        else:
            absdiff = abs(a-b)
            if absdiff.max() > tol:
                loc = np.unravel_index(absdiff.argmax(), absdiff.shape)
                print('a')
                print(a)
                print()
                print('b')
                print(b)
                print()
                print('Absolute difference')
                print(absdiff)
                self.fail('Maximum abs difference between array elements is %e at location %r' %
                          (absdiff.max(), loc))

    def assertAtomsAlmostEqual(self, a, b, tol=1e-7):
        self.assertArrayAlmostEqual(a.positions, b.positions, tol)
        self.assertArrayAlmostEqual(a.numbers, b.numbers)
        self.assertArrayAlmostEqual(a._cell, b._cell)
        self.assertArrayAlmostEqual(a._pbc, b._pbc)

def skip(f):
    """
    Decorator which can be used to skip unit tests
    """
    def g(self):
        logging.warning('skipping test %s' % f.__name__)
    return g
