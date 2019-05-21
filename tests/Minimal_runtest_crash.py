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
"""

Only on macosx,

This thing crashes when being run with runtests but not when run as a normal MPI_process

"""

import pytest
from FFTEngine import PFFTEngine
import numpy as np
from PyCo.Topography import NPYReader


def test_crashes(comm):
    fn = "worflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)
    if comm.Get_rank() == 0:
        np.save(fn, data)
    comm.barrier()

    # These two lines together produce strange error
    fileReader = NPYReader(fn, comm=comm)
    fftengine = PFFTEngine(res, comm=comm)

    # Calling them in the other order produces the error as well
    #fileReader = NPYReader(fn, comm=comm)
    #fftengine = PFFTEngine(res, comm=comm)

if __name__ == "__main__":
    # running this with a normal mpirun is ok
    print("main run")
    from mpi4py import MPI
    test_crashes(MPI.COMM_WORLD)