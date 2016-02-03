#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   01-ImportTest.py

@author Till Junge <till.junge@kit.edu>

@date   27 Jan 2015

@brief  Import testing

@section LICENCE

 Copyright (C) 2015 Till Junge

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
