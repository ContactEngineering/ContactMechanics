#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @file   FromFile.py
#
# @author Till Junge <till.junge@kit.edu>
#
# @date   26 Jan 2015
#
# @brief  Surface profile from file input
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

import numpy as np
import os

from . import NumpySurface

class NumpyTxtSurface(NumpySurface):
    """ Reads a surface profile from file and presents in in a Surface-
        conformant manner.
    """
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
        super().__init__(np.loadtxt(fname))



if __name__ == '__main__':
    import tempfile, os
    with tempfile.TemporaryDirectory() as folder:
        a = np.random.random((16, 32))
        fname = os.path.join(folder, "tmpfile")
        np.savetxt(fname, a)
        surf = NumpyTxtSurface(fname)
        print(surf.dim)
        print(surf.resolution)
