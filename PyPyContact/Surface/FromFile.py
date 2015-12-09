#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   FromFile.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Surface profile from file input

@section LICENCE

 Copyright (C) 2015 Till Junge

PyPyContact is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

PyPyContact is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import numpy as np
import os
import re

from . import NumpySurface


class NumpyTxtSurface(NumpySurface):
    """ Reads a surface profile from file and presents in in a Surface-
        conformant manner.
    """
    # pylint: disable=too-few-public-methods
    name = 'surface_from_np_file'

    def __init__(self, fobj, size=None, factor=1.):
        """
        Keyword Arguments:
        fobj -- filename or file object
        """
        if not hasattr(fobj, 'read'):
            if not os.path.isfile(fobj):
                zfobj = fobj + ".gz"
                if os.path.isfile(zfobj):
                    fobj = zfobj
                else:
                    raise FileNotFoundError(
                        "No such file or directory: '{}(.gz)'".format(
                            fobj))
            self.fname = fobj
        self.fobj = fobj
        super().__init__(factor*np.loadtxt(fobj), size=size)
read_matrix = NumpyTxtSurface

class NumpyAscSurface(NumpySurface):
    """ Reads a surface profile from an asc file and presents it in a Surface-
        conformant manner.
    """
    name = 'surface_from_asc_file'
    _units = {'m': 1.0, 'mm': 1e-3, 'um': 1e-6, 'nm': 1e-9, 'A': 1e-10}

    def __init__(self, fobj, unit=None, x_unit=1.0, z_unit=1.0):
        """
        Keyword Arguments:
        fobj -- filename or file object
        """
        if not hasattr(fobj, 'read'):
            if not os.path.isfile(fobj):
                raise FileNotFoundError(
                    "No such file or directory: '{}(.gz)'".format(
                        fobj))
            self.fname = fobj
            fobj = open(self.fname)
        self.fobj = fobj
        profile, size = self.load(unit)
        super().__init__(profile*z_unit, size=tuple((x_unit*s for s in size)))

    def load(self, unit):
        """ read in a surface file
        """
        checks = list()
        checks.append((re.compile("x-pixels = ([0-9]+)"), int, "x_res"))
        checks.append((re.compile("y-pixels = ([0-9]+)"), int, "y_res"))
        checks.append((re.compile("x-length = ([0-9.]+)"), float, "x_siz"))
        checks.append((re.compile("y-length = ([0-9.]+)"), float, "y_siz"))
        checks.append((re.compile(r"x-unit = (\w+)"), str, "x_unit"))
        checks.append((re.compile(r"y-unit = (\w+)"), str, "y_unit"))
        checks.append((re.compile(r"z-unit = (\w+)"), str, "z_unit"))

        data = None
        xres = yres = xsiz = ysiz = xunit = yunit = zunit = None

        def process_comment(line):
            "Find and interpret known comments in the header of the asc file"
            def check(line, reg, fun):
                "Check whether line fits a known comment syntax"
                match = reg.search(line)
                if match is not None:
                    return fun(match.group(1))
                return None
            nonlocal xres, yres, xsiz, ysiz, xunit, yunit, zunit, data
            matches = {key: check(line, reg, fun)
                       for (reg, fun, key) in checks}
            if matches['x_res'] is not None:
                xres = matches['x_res']
            elif matches['y_res'] is not None:
                yres = matches['y_res']
            elif matches['x_siz'] is not None:
                xsiz = matches['x_siz']
            elif matches['y_siz'] is not None:
                ysiz = matches['y_siz']
            elif matches['x_unit'] is not None:
                xunit = matches['x_unit']
            elif matches['y_unit'] is not None:
                yunit = matches['y_unit']
            elif matches['z_unit'] is not None:
                zunit = matches['z_unit']
            if xres is not None and yres is not None:
                data = np.zeros((xres, yres))

        row_nu = 0
        with self.fobj as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    process_comment(line)
                else:
                    data[row_nu, :] = line.strip().split()
                    row_nu += 1
        if row_nu != xres:
            raise Exception(
                ("The number of rows read from the file '{}' does not match "
                 "the declared resolution").format(self.fname))

        # Handle units -> convert to target unit
        if xunit is None and zunit is not None:
            xunit = zunit
        if yunit is None and zunit is not None:
            yunit = zunit

        if unit is not None:
            if xunit is not None:
                xsiz *= self._units[xunit]/self._units[unit]
            if yunit is not None:
                ysiz *= self._units[yunit]/self._units[unit]
            if zunit is not None:
                data *= self._units[zunit]/self._units[unit]

        return data, (xsiz, ysiz)
read_asc = NumpyAscSurface

def read_xyz(fn):
    x, y, z = np.loadtxt(fn, unpack=True)
    
    # Sort x-values into bins. Assume that points on surface are equally spaced.
    dx = x[1]-x[0]
    binx = np.array(x/dx+0.5, dtype=int)
    n = np.bincount(binx)
    ny = n[0]
    assert np.all(n == ny)

    # Sort y-values into bins.
    dy = y[binx==0][1]-y[binx==0][0]
    biny = np.array(y/dy+0.5, dtype=int)
    n = np.bincount(biny)
    nx = n[0]
    assert np.all(n == nx)

    # Sort data into bins.
    data = np.zeros((nx, ny))
    data[binx, biny] = z

    # Sanity check. Should be covered by above asserts.
    value_present = np.zeros((nx, ny), dtype=bool)
    value_present[binx, biny] = True
    assert np.all(value_present)

    return NumpySurface(data, size=(dx*nx, dy*ny))
