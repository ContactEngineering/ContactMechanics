#
# Copyright 2017, 2020 Lars Pastewka
#           2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import sys

import numpy as np

from SurfaceTopography.Uniform.GeometryAnalysis import assign_segment_numbers_area
from ContactMechanics.IO.NetCDF import NetCDFContainer

nc = NetCDFContainer(sys.argv[1])
frame = nc[int(sys.argv[2])]

f = frame.forces
c = f > 1e-6

# Identify individual segments
ns, s = assign_segment_numbers_area(c)
assert s.max() == ns

# Get segment length
sl = np.bincount(np.ravel(s))
drep = sl[1:].mean()
print('drep =', drep)
