
import os

import numpy as np
import pytest

from NuMPI import MPI

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import Topography
from PyCo.Topography.IO import read_topography
from PyCo.Topography.IO.NC import NCReader
from PyCo.Topography.Generation import fourier_synthesis

def test_save_and_load(comm):
    nb_grid_pts = (128, 128)
    size = (3, 3)

    np.random.seed(1)
    t = fourier_synthesis(nb_grid_pts, size, 0.8, rms_slope=0.1)

    substrate = PeriodicFFTElasticHalfSpace(nb_grid_pts, 1, fft='mpi', communicator=comm)
    dt = t.domain_decompose(substrate.subdomain_locations, substrate.nb_subdomain_grid_pts, comm)

    # Attempt to open full file on each process
    dt.to_netcdf('parallel_save_test.nc')

    t2 = read_topography('parallel_save_test.nc')

    assert t.physical_sizes == t2.physical_sizes
    np.testing.assert_array_almost_equal(t.heights(), t2.heights())

    # Attempt to open file in parallel
    r = NCReader('parallel_save_test.nc', communicator=comm)
    t3 = r.topography(subdomain_locations=substrate.subdomain_locations,
                      nb_subdomain_grid_pts=substrate.nb_subdomain_grid_pts)

    np.testing.assert_array_almost_equal(dt.heights(), t3.heights())

    assert t3.is_periodic


if __name__ == '__main__':
    test_save_and_load(MPI.COMM_WORLD)
