#
# Copyright 2018-2019 Lars Pastewka
#           2018 Antoine Sanner
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
Tests the pylint (and possibly pep8) conformity of the code
"""
import unittest
from pylint import epylint
import pep8

import PyCo

class SystemTest(unittest.TestCase):
    def setUp(self):
        self.modules = list([PyCo,
                             PyCo.ContactMechanics,
                             PyCo.ContactMechanics.FFTElasticHalfSpace,
                             PyCo.ContactMechanics.Substrates,
                             PyCo.SurfaceTopography,
                             PyCo.SurfaceTopography.FromFile,
                             PyCo.SurfaceTopography.HeightContainer,
                             PyCo.System,

                             PyCo.ContactMechanics.Systems,
                             PyCo.Tools,
                             PyCo.Tools.Optimisation.AugmentedLagrangian,
                             PyCo.Tools.Optimisation.NewtonConfidenceRegion,
                             PyCo.Tools.Optimisation.NewtonLineSearch,
                             PyCo.Tools.Optimisation.common,
                             PyCo.Tools.common,
                             PyCo.Goodies,
                             PyCo.Goodies.SurfaceAnalysis,
                             PyCo.Goodies.SurfaceGeneration])

    def te_st_pylint_bitchiness(self):
        print()
        options = ' --rcfile=tests/pylint.rc --disable=locally-disabled'
        for module in self.modules:
            epylint.py_run(module.__file__ + options)

    def te_st_pep8_conformity(self):
        print()
        pep8style = pep8.StyleGuide()
        pep8style.check_files((mod.__file__ for mod in self.modules))
