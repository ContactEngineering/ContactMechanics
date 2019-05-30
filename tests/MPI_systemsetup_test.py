#
# Copyright 2018-2019 Antoine Sanner
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

represents the UseCase of creating System with MPI parallelization

"""

import pytest

from mpi4py import MPI
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace,PeriodicFFTElasticHalfSpace
from PyCo.System.Factory import make_system
from PyCo.ContactMechanics.Interactions import HardWall
from PyCo.Topography.IO import NPYReader, open_topography
import numpy as np
import os

DATADIR = os.path.dirname(os.path.realpath(__file__))

# @pytest.fixture
# def examplefile(comm):
#     fn = DATADIR + "/worflowtest.npy"
#     res = (128,64)
#     np.random.seed(1)
#     data  = np.random.random(res )
#     data -= np.mean(data)
#     if comm.Get_rank == 0:
#         np.save(fn, data)
#
#     comm.barrier()
#     return (fn, res, data)

@pytest.mark.parametrize("HS", [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
@pytest.mark.parametrize("loader", [open_topography, NPYReader])
def test_LoadTopoFromFile(comm, fftengine_type, HS, loader):

    fn = DATADIR + "/worflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)
    if comm.Get_rank() == 0:
        np.save(fn, data)
    comm.barrier() # all processors wait on the file to be created

    #fn, res, data = examplefile
    interaction = HardWall()

    # Read metadata from the file and returns a UniformTopgraphy Object
    fileReader = loader(fn, comm=comm)

    #pdb.set_trace()

    assert fileReader.nb_grid_pts == res

    # create a substrate according to the topography
    fftengine = fftengine_type(fileReader.nb_grid_pts, comm = comm)
    Es = 1
    if fileReader.physical_sizes is not None:
        substrate = HS(nb_grid_pts=fileReader.nb_grid_pts, size=fileReader.physical_sizes, young=Es, fftengine=fftengine)
    else:
        substrate = HS(nb_grid_pts=fileReader.nb_grid_pts,size=fileReader.nb_grid_pts, young = Es, fftengine=fftengine )

    top = fileReader.topography(substrate)

    assert top.nb_grid_pts == substrate.nb_grid_pts
    assert top.nb_subdomain_grid_pts == substrate.topography_nb_subdomain_grid_pts
          # or top.nb_subdomain_grid_pts == (0,0) # for FreeFFTElHS
    assert top.subdomain_locations == substrate.topography_subdomain_locations

    np.testing.assert_array_equal(top.heights(),data[top.subdomain_slices])

    # test that the slicing is what is expected

    fulldomain_field = np.arange(np.prod(substrate.nb_domain_grid_pts)).reshape(substrate.nb_domain_grid_pts)

    np.testing.assert_array_equal(fulldomain_field[top.subdomain_slices],fulldomain_field[tuple([slice(substrate.subdomain_locations[i],substrate.subdomain_locations[i]+max(0,min(substrate.nb_grid_pts[i] - substrate.subdomain_locations[i],substrate.nb_subdomain_grid_pts[i]))) for i in range(substrate.dim)])])

    # Test Computation of the rms_height
    # Sq
    assert top.rms_height(kind="Sq") == np.sqrt(np.mean((data - np.mean(data))**2))
    #Rq
    assert top.rms_height(kind="Rq") == np.sqrt(np.mean((data - np.mean(data,axis = 0))**2))

    system = make_system(substrate, interaction, top)

        # make some tests on the system

@pytest.mark.xfail(run=False)
def test_make_system_from_file():
    """
    longtermgoal for confortable and secure use
    Returns
    -------

    """
    # Maybe it will be another Function or class

    substrate =  PeriodicFFTElasticHalfSpace
    interaction = HardWall

    system = make_system(substrate, interaction, fn)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    fn = "worflowtest.npy"
    res = (128, 64)
    np.random.seed(1)
    data = np.random.random(res)
    data -= np.mean(data)

    np.save(fn, data)
    test_LoadTopoFromFile(comm, (fn, res, data), HS=PeriodicFFTElasticHalfSpace,
                              loader=NPYReader)