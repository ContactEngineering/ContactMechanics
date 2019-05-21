
import pytest
import numpy as np
import os

from PyCo.Topography.IO.NPY import NPYReader
from PyCo.Topography.IO.NPY import save_npy
from PyCo.Topography import open_topography

from NuMPI.Tools import Reduction

import pytest
from NuMPI import MPI
pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="tests only serial funcionalities, please execute with pytest")

def test_save_and_load(comm_self, file_format_examples):
    print("Hello")
    # sometimes the surface isn't transposed the same way when

    topography = open_topography(
        os.path.join(file_format_examples, 'example4.di'), format="di").topography()

    npyfile = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "test_save_and_load.npy")
    print(npyfile)
    topography.pnp = Reduction(comm_self) # save_npy extracts the communicator
    # from the pnp
    save_npy(npyfile, topography)

    class dummy_substrate:
        resolution = topography.resolution
        topography_subdomain_resolution = topography.resolution
        topography_subdomain_location = (0,0)
        size= (1., 1.)
        pnp = topography.pnp

    loaded_topography = NPYReader(npyfile, comm=comm_self
                                  ).topography(substrate=dummy_substrate)

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())

    os.remove(npyfile)

@pytest.mark.xfail
def test_save_and_load_np(comm_self, file_format_examples):

    # sometimes the surface isn't transposed the same way when

    topography = open_topography(
        os.path.join(file_format_examples, 'example4.di'), format="di").topography()

    npyfile = "test_save_and_load_np.npy"
    np.save(npyfile, topography.heights())

    loaded_topography = NPYReader(npyfile, comm=comm_self).topography(size=(1., 1.))

    np.testing.assert_allclose(loaded_topography.heights(), topography.heights())
    
    os.remove(npyfile)
