#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   01-ImportTest.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Import testing

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

try:
    import unittest
    import importlib
    import numpy as np
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class ImportabilityChecks(unittest.TestCase):

    def import_module(self, module):
        return_code = -1
        try:
            importlib.import_module(module)
            return_code = 0
        except ImportError: pass
        return return_code

    def test_PyCo(self):
        self.assertEqual(self.import_module("PyCo"), 0)

    def test_ContactMechanics(self):
        self.assertEqual(self.import_module("PyCo.ContactMechanics"), 0)

    def test_SolidMechanics(self):
        self.assertEqual(self.import_module("PyCo.SolidMechanics"), 0)

    def test_Solver(self):
        self.assertEqual(self.import_module("PyCo.System"), 0)

    def test_Surface(self):
        self.assertEqual(self.import_module("PyCo.Surface"), 0)

if __name__ == '__main__':
    unittest.main()
