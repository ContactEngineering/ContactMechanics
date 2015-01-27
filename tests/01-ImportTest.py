#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   01-ImportTest.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   27 Jan 2015
#
# @brief  Import testing
#
# @section LICENCE
#
#  Copyright (C) 2015 Till Junge
#
# PyPyContact is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# PyPyContact is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

import unittest
import importlib

class ImportabilityChecks(unittest.TestCase):

    def import_module(self, module):
        return_code = -1
        try:
            importlib.import_module(module)
            return_code = 0
        except ImportError: pass
        return return_code

    def test_PyPyContact(self):
        self.assertEqual(self.import_module("PyPyContact"), 0)

    def test_ContactMechanics(self):
        self.assertEqual(self.import_module("PyPyContact.ContactMechanics"), 0)

    def test_SolidMechanics(self):
        self.assertEqual(self.import_module("PyPyContact.SolidMechanics"), 0)

    def test_Solver(self):
        self.assertEqual(self.import_module("PyPyContact.Solver"), 0)

    def test_Surface(self):
        self.assertEqual(self.import_module("PyPyContact.Surface"), 0)

if __name__ == '__main__':
    unittest.main()
