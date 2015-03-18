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

    def __init__(self, fname):
        """
        Keyword Arguments:
        fname -- filename
        """
        if not os.path.isfile(fname):
            zfname = fname + ".gz"
            if os.path.isfile(zfname):
                fname = zfname
            else:
                raise FileNotFoundError(
                    "No such file or directory: '{}(.gz)'".format(
                        fname))
        self.fname = fname
        super().__init__(np.loadtxt(fname))


class NumpyAscSurface(NumpySurface):
    """ Reads a surface profile from an asc file and presents in in a Surface-
        conformant manner.
    """
    name = 'surface_from_nc_file'

    def __init__(self, fname, x_unit=1e-6, z_unit=1e-9):
        """
        Keyword Arguments:
        fname -- filename
        """
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                "No such file or directory: '{}(.gz)'".format(
                    fname))
        self.fname = fname
        profile, size = self.load()
        super().__init__(profile*z_unit, tuple((x_unit*s for s in size)))

    def load(self, ):
        """ read in a surface file
        """
        checks = list()
        checks.append((re.compile("x-pixels = ([0-9]+)"), int, "x_res"))
        checks.append((re.compile("y-pixels = ([0-9]+)"), int, "y_res"))
        checks.append((re.compile("x-length = ([0-9.]+)"), float, "x_siz"))
        checks.append((re.compile("y-length = ([0-9.]+)"), float, "y_siz"))

        data = None
        xres = yres = xsiz = ysiz = None

        def process_comment(line):
            def check(line, reg, fun):
                match = reg.search(line)
                if match is not None:
                    return fun(match.group(1))
                return None
            nonlocal xres, yres, xsiz, ysiz, data
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
            if xres is not None and yres is not None:
                data = np.zeros((xres, yres))

        row_nu = 0
        with open(self.fname) as fh:
            for line in fh:
                if line.startswith("#"):
                    process_comment(line)
                else:
                    data[row_nu, :] = line.strip().split()
                    row_nu += 1
        if row_nu != xres:
            raise Exception(
                ("The number of rows read from the file '{}' does not match "
                 "the declared resolution").format(self.fname))
        return data, (xsiz, ysiz)
