#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   07-ConformityTest.py

@author Till Junge <till.junge@kit.edu>

@date   23 Feb 2015

@brief  Tests the pylint (and possibly pep8) conformity of the code

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
    from pylint import epylint
    import pep8

    import PyCo
except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

class SystemTest(unittest.TestCase):
    def setUp(self):
        self.modules = list([PyCo,
                             PyCo.ContactMechanics,
                             PyCo.ContactMechanics.Interactions,
                             PyCo.ContactMechanics.Lj93,
                             PyCo.ContactMechanics.VdW82,
                             PyCo.ContactMechanics.Potentials,
                             PyCo.SolidMechanics,
                             PyCo.SolidMechanics.FFTElasticHalfSpace,
                             PyCo.SolidMechanics.Substrates,
                             PyCo.Surface,
                             PyCo.Surface.FromFile,
                             PyCo.Surface.SurfaceDescription,
                             PyCo.System,
                             PyCo.System.SmoothSystemSpecialisations,
                             PyCo.System.Systems,
                             PyCo.Tools,
                             PyCo.Tools.Optimisation.AugmentedLagrangian,
                             PyCo.Tools.Optimisation.NewtonConfidenceRegion,
                             PyCo.Tools.Optimisation.NewtonLineSearch,
                             PyCo.Tools.Optimisation.common,
                             PyCo.Tools.common,
                             PyCo.Goodies,
                             PyCo.Goodies.SurfaceAnalysis,
                             PyCo.Goodies.SurfaceGeneration])

    def test_pylint_bitchiness(self):
        print()
        options = ' --rcfile=tests/pylint.rc --disable=locally-disabled'
        for module in self.modules:
            epylint.py_run(module.__file__ + options)

    def test_pep8_conformity(self):
        print()
        pep8style = pep8.StyleGuide()
        pep8style.check_files((mod.__file__ for mod in self.modules))
