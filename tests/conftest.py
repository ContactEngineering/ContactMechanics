#
# Copyright 2019 Lars Pastewka
#           2019 Antoine Sanner
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
import os

from runtests.mpi import MPITestFixture

from NuMPI import MPI
from muFFT import FFT

comm = MPITestFixture([1, 2, 4], scope='session')
comm_self = MPITestFixture([1], scope='session')

maxcomm = MPITestFixture([MPI.COMM_WORLD.Get_size()], scope="session")


@pytest.fixture(scope="session")
def file_format_examples():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'file_format_examples')


@pytest.fixture(scope="session")
def fftengine_type(comm):
    try:
        fftname = "mpi"
        engine = FFT((2 * comm.Get_size(), 3 * comm.Get_size()), fft="mpi", communicator=comm)
    except:
        if comm.Get_size() == 1:
            fftname = "fftw"
        else:
            raise ValueError("No appropriate fft library detected")
    return fftname
