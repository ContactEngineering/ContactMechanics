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