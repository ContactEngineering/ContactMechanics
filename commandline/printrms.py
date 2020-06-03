#
# Copyright 2016-2018, 2020 Lars Pastewka
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

"""
Command line front-end for printing statistical properties to screen
"""

from argparse import ArgumentParser, ArgumentTypeError

import numpy as np
import matplotlib.pyplot as plt

from SurfaceTopography.FromFile import read, detect_format

###

parser = ArgumentParser(description='Print statistical properties to screen.')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')
arguments = parser.parse_args()

###

surfaces = read(arguments.filename, format=detect_format(arguments.filename))

if type(surfaces) is not list:
    surfaces = [surfaces]

###

for surface in surfaces:
    try:
        print('---', surface.info['data'], '---')
    except:
        print('---')
    print('RMS height =', surface.rms_height(), surface.unit)
    print('RMS slope = ', surface.rms_slope())
    print('RMS curvature =', surface.rms_curvature(),
          '{}^-1'.format(surface.unit))
