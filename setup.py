#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   setup.py

@author Till Junge <till.junge@kit.edu>

@date   26 Jan 2015

@brief  Installation script

@section LICENCE

Copyright 2015-2017 Till Junge, Lars Pastewka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import versioneer
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os
import sys


# Hack for a --openmp option that compiles with parallel FFTW3
if '--openmp' in sys.argv:
    index = sys.argv.index('--openmp')
    sys.argv.pop(index)  # Removes the '--openmp'
    extra_compile_args = ["-std=c++11", "-fopenmp"]
    extra_link_args = ["-lfftw3_omp", "-lfftw3", "-lm", "-fopenmp"]
else:
    extra_compile_args = ["-std=c++11"]
    extra_link_args = ["-lfftw3", "-lm"]


extensions = [
    Extension(
        name='_PyCo',
        sources=['c/patchfinder.cpp'],
        include_dirs=[np.get_include()]
        ),
    Extension(
        name="PyCo.Tools.fftext",
        sources=["PyCo/Tools/fftext.pyx", "PyCo/Tools/fftext_cc.cc"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
        language="c++"),
    Extension(
        name="PyCo.Tools.Optimisation.ConstrainedConjugateGradientsOpt",
        sources=["PyCo/Tools/Optimisation/ConstrainedConjugateGradientsOpt.pyx"],
        include_dirs=[np.get_include()],
        language="c++"),
    Extension(
        name="PyCo.Goodies.ScanningProbe",
        sources=["PyCo/Goodies/ScanningProbe.pyx"],
        include_dirs=[np.get_include()],
        language="c++")]



setup(
    name = "PyCo",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    packages = find_packages(),
    package_data = {'': ['ChangeLog.md']},
    include_package_data = True,
    ext_modules = cythonize(extensions),
    # metadata for upload to PyPI
    author = "Lars Pastewka",
    author_email = "lars.pastewka@imtek.uni-freiburg.de",
    description = "Efficient contact mechanics with Python",
    license = "MIT",
    test_suite = 'tests'
)
