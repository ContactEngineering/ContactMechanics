#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   setup.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Installation script

@section LICENCE

 Copyright (C) 2015 Till Junge

This project is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

This project is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import versioneer
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

fftext_path = "PyCo/Tools/"
extensions = [
    Extension(
        name="PyCo.Tools.fftext",
        sources=[fftext_path + src_name for src_name in ("fftext.pyx", "fftext_cc.cc")],
        extra_compile_args=["-std=c++1y", "-fopenmp"],
        extra_link_args=["-lfftw3_omp", "-lfftw3", "-lm"],
        language="c++"),]



setup(
    name = "PyCo",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    packages = find_packages(),
    ext_modules=extensions,
    # metadata for upload to PyPI
    author = "Till Junge",
    author_email = "till.junge@kit.edu",
    description = "Simple contact mechanics code",
    license = "GPLv3",
    test_suite = 'tests'
)
