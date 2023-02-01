#
# Copyright 2022 Lars Pastewka
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

import os
import subprocess


class CannotDiscoverVersion(Exception):
    pass


def get_version_from_git():
    """
    Discover version from git repository.
    """
    if not os.path.exists('.git'):
        raise CannotDiscoverVersion('.git subdirectory does not exist.')

    try:
        git_describe = subprocess.run(
            ['git', 'describe', '--tags', '--dirty', '--always'],
            stdout=subprocess.PIPE)
    except FileNotFoundError:
        git_describe = None
    if git_describe is None or git_describe.returncode != 0:
        raise CannotDiscoverVersion('git execution failed.')
    version = git_describe.stdout.decode('latin-1').strip()

    dirty = version.endswith('-dirty')

    # Make version PEP 440 compliant
    if dirty:
        version = version.replace('-dirty', '')
    version = version.strip('v')  # Remove leading 'v' if it exists
    version = version.replace('-', '.dev', 1)
    version = version.replace('-', '+', 1)
    if dirty:
        version += '.dirty'

    return version


_pkg_name = __name__.replace('.DiscoverVersion', '')

# importlib is present in Python >= 3.8
try:
    from importlib.metadata import version

    __version__ = version(_pkg_name)
except ImportError:
    __version__ = None

# git works if we are in the source repository
if __version__ is None:
    try:
        __version__ = get_version_from_git()
    except CannotDiscoverVersion:
        __version__ = None

# Nope. Out of options.

if __version__ is None:
    raise CannotDiscoverVersion('Tried importlib, pkg_resources, PKG-INFO and git')
