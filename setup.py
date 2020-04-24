#
# Copyright 2016, 2019 Lars Pastewka
#           2018-2019 Antoine Sanner
#           2015-2016 Till Junge
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

import re

import versioneer
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):
        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)


#
# Optain LAPACK configuration from numpy
#

try:
    import numpy as np
except ImportError:
    print("WARNING: Could not find numpy, skipping LAPACK detection.")
    np = None

extra_link_args = []
if np is not None:
    for k, v in np.__config__.__dict__.items():
        if re.match('lapack_.*_info', k):
            if v:
                print("* Using LAPACK information from '%s' dictionary in " \
                    "numpy.__config__" % k)
                try:
                    print("    library_dirs = '%s'" % v['library_dirs'])
                    lib_dirs += v['library_dirs']
                except:
                    print("    No 'library_dirs' entry found.")
                try:
                    print("    libraries = '%s'" % v['libraries'])
                    libs += v['libraries']
                except:
                    print("    No 'libraries' entry found.")
                try:
                    print("    extra_link_args = '%s'" % v['extra_link_args'])
                    extra_link_args += v['extra_link_args']
                except:
                    print("    No 'extra_link_args' entry found.")
                # We use whichever lapack_*_info comes first, hopefully there is
                # just one.
                break


extra_compile_args = ["-std=c++11"]

scripts = ['commandline/hard_wall.py',
           'commandline/soft_wall.py',
           'commandline/plotacf.py',
           'commandline/plotpsd.py',
           'commandline/plotmap.py']

extensions = [
    Extension(
        name='_PyCo',
        sources=['c/autocorrelation.c',
                  'c/bicubic.cpp',
                  'c/patchfinder.cpp',
                  'c/PyCo_module.cpp'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name="PyCo",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(cmdclass={'build_ext': CustomBuildExtCommand}),
    scripts=scripts,
    packages=find_packages(),
    package_data={'': ['ChangeLog.md']},
    include_package_data=True,
    ext_modules=extensions,
    # metadata for upload to PyPI
    author="Lars Pastewka",
    author_email="lars.pastewka@imtek.uni-freiburg.de",
    description="Efficient contact mechanics with Python",
    license="MIT",
    test_suite='tests',
    # dependencies
    python_requires='>3.5.0',
    install_requires=[
        'numpy>=1.11.0',
        'NuMPI',
        'muFFT'
    ]
)
