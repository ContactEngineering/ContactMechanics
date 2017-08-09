#! /usr/bin/env python3

import sys

import numpy as np

from PyCo.Tools.NetCDF import NetCDFContainer
from PyCo.Tools.ContactAreaAnalysis import assign_segment_numbers

###

nc = NetCDFContainer(sys.argv[1])
frame = nc[int(sys.argv[2])]

f = frame.forces
c = f > 1e-6

# Indentify indidual segments
ns, s = assign_segment_numbers(c)
assert s.max() == ns

# Get segment length
sl = np.bincount(np.ravel(s))
drep = sl[1:].mean()
print('drep =', drep)