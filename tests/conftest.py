#
# Copyright 2019 Antoine Sanner
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

import pytest

collect_ignore_glob = ["*MPI_*.py"]

from runtests.mpi import MPITestFixture

#@pytest.fixture
#def commsizes(request):
#    return [int(s) for s in request.config.getoption("--commsizes").split(',')]

#comm = MPITestFixture([int(s) for s in config.getoption("--commsizes").split(',')], scope='session')
comm = MPITestFixture([1,2,4], scope='session')

@pytest.fixture(scope="session")
def fftengine_class(comm):
    try:
        from FFTEngine import PFFTEngine as engine
    except Exception as err:
        if comm.Get_size() == 1:
            try:
                from FFTEngine import FFTWEngine as engine
            except:
                from FFTEngine import NumpyFFTEngine as engine
        else:
            raise err
    return engine