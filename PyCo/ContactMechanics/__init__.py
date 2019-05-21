#
# Copyright 2019 Antoine Sanner
#           2016 Lars Pastewka
#           2016 Till Junge
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
Defines all interaction modes used in PyCo
"""

from .Interactions import Interaction, HardWall, SoftWall
from .Potentials import Potential, SmoothPotential
from .Potentials import LinearCorePotential, ParabolicCutoffPotential, MinimisationPotential

from .Adhesion import ExpPotential
from .Harmonic import HarmonicPotential
from .Lj93 import LJ93, LJ93smooth, LJ93smoothMin, LJ93SimpleSmooth, LJ93SimpleSmoothMin
from .VdW82 import VDW82, VDW82smooth, VDW82smoothMin, VDW82SimpleSmooth, VDW82SimpleSmoothMin
