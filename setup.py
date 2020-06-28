#
# Copyright 2016, 2020 Lars Pastewka
#           2018, 2020 Antoine Sanner
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

from setuptools import setup, find_packages


scripts = [
   'commandline/hard_wall.py',
   'commandline/plotacf.py',
   'commandline/plotpsd.py',
   'commandline/plotmap.py'
   ]

setup(
    name="ContactMechanics",
    scripts=scripts,
    packages=find_packages(),
    package_data={'': ['ChangeLog.md']},
    include_package_data=True,
    # metadata for upload to PyPI
    author="Lars Pastewka",
    author_email="lars.pastewka@imtek.uni-freiburg.de",
    description="Efficient contact mechanics using elastic half-space methods",
    license="MIT",
    test_suite='test',
    # dependencies
    python_requires='>=3.5.0',
    use_scm_version=True,
    zip_safe=True,
    setup_requires=[
        'setuptools_scm>=3.5.0'
    ],
    install_requires=[
        'numpy>=1.11.0',
        'NuMPI>=0.1.2',
        'muFFT>=0.9.3',
        'SurfaceTopography>=0.90.0'
    ]
)
