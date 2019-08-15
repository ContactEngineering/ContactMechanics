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
from PyCo.ContactMechanics import HardWall, VDW82
from PyCo.Topography import make_sphere
from PyCo.Topography.IO import NPYReader, open_topography


import numpy as np

import os
from NuMPI.Tools import Reduction

DATADIR = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture
def examplefile(comm):
    fn = DATADIR + "/worflowtest.npy"
    res = (128,64)
    np.random.seed(1)
    data  = np.random.random(res )
    data -= np.mean(data)
    if comm.rank == 0:
        np.save(fn, data)

    comm.barrier()
    return (fn, res, data)

#DATAFILE = DATADIR + "/worflowtest.npy"
#@pytest.fixture
#def data(comm):
#    res = (256,256)#(128, 64)
#    np.random.seed(1)
#    data = np.random.random(res)
#    data -= np.mean(data)
#    if comm.Get_rank() == 0:
#        np.save(DATAFILE, data)
#    comm.barrier() # all processors wait on the file to be created
#    return data

@pytest.mark.parametrize("HS", [PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace])
@pytest.mark.parametrize("loader", [open_topography, NPYReader])
def test_LoadTopoFromFile(comm, fftengine_type, HS, loader, examplefile):
    #fn = DATAFILE
    fn, res, data = examplefile
    interaction = HardWall()

    # Read metadata from the file and returns a UniformTopgraphy Object
    fileReader = loader(fn, communicator=comm)
    fileReader.nb_grid_pts = fileReader._nb_grid_pts #FIXME: workaround

    assert fileReader.nb_grid_pts == res

    Es = 1
    substrate = HS(nb_grid_pts=fileReader.nb_grid_pts,
                   physical_sizes=fileReader.nb_grid_pts,
                   young = Es, fft="fftwmpi",
                   communicator=comm )

    top = fileReader.topography(
        subdomain_locations=substrate.topography_subdomain_locations,
        nb_subdomain_grid_pts=substrate.topography_nb_subdomain_grid_pts,
        physical_sizes=substrate.physical_sizes)

    assert top.nb_grid_pts == substrate.nb_grid_pts
    assert top.nb_subdomain_grid_pts \
           == substrate.topography_nb_subdomain_grid_pts
          # or top.nb_subdomain_grid_pts == (0,0) # for FreeFFTElHS
    assert top.subdomain_locations == substrate.topography_subdomain_locations

    np.testing.assert_array_equal(top.heights(),data[top.subdomain_slices])

    # test that the slicing is what is expected

    fulldomain_field = np.arange(np.prod(substrate.nb_domain_grid_pts)
                                 ).reshape(substrate.nb_domain_grid_pts)

    np.testing.assert_array_equal(
        fulldomain_field[top.subdomain_slices],
        fulldomain_field[tuple([
            slice(substrate.subdomain_locations[i],
            substrate.subdomain_locations[i]
                  + max(0,min(substrate.nb_grid_pts[i]
                  - substrate.subdomain_locations[i],
            substrate.nb_subdomain_grid_pts[i])))
            for i in range(substrate.dim)])])

    # Test Computation of the rms_height


    ############## BEGINDEBUG
    assert top.rms_height(kind="Sq") \
           == np.sqrt(np.mean((data - np.mean(data))**2))

    #Rq
    assert top.rms_height(kind="Rq") \
           == np.sqrt(np.mean((data - np.mean(data,axis = 0))**2))

    system = make_system(substrate, interaction, top)

    # make some tests on the system


def test_make_system_from_file(examplefile, comm):
    """
    longtermgoal for confortable and secure use
    Returns
    -------

    """
    # TODO: test this on npy and nc file
    # Maybe it will be another Function or class
    fn, res, data = examplefile

    substrate =  PeriodicFFTElasticHalfSpace
    interaction = HardWall()

    system = make_system(substrate="periodic",
                         interaction=interaction,
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20.,30.),
                         young=1)

    print( system.__class__)

def test_make_system_from_file_serial(comm_self):
    """
    same as test_make_system_from_file but with the reader being not MPI
    compatible
    Returns
    -------

    """
    pass

#def test_automake_substrate(comm):
#    surface = make_sphere(2, (4,4), (1., 1.), )

def test_smoothcontactsystem_no_minimize_proxy(examplefile, comm):
    """
    asserts the smoothcontactsystem doesn't allow the use of minimize_proxy for
    distributed computations
    """
    fn, res, data = examplefile

    interaction = VDW82(1.,1., communicator=comm)

    if comm.size ==1:
        system = make_system(substrate="periodic",
                             interaction=interaction,
                             surface=fn,
                             communicator=comm,
                             physical_sizes=(20., 30.),
                             young=1)
    else:

        with pytest.raises(ValueError):
            system = make_system(substrate="periodic",
                                 interaction=interaction,
                                 surface=fn,
                                 communicator=comm,
                                 physical_sizes=(20., 30.),
                                 young=1)
            print(system.surface.is_domain_decomposed)
            system.minimize_proxy()


def test_make_free_system(examplefile, comm):
    """
    For number of processors > 1 it SmartSmoothContactSystem
    doesn't work.
    Parameters
    ----------
    comm

    Returns
    -------

    """
    fn, res, data = examplefile

    interaction = VDW82(1., 1., communicator=comm)
    system = make_system(substrate="free",
                         interaction=interaction,
                         surface=fn,
                         communicator=comm,
                         physical_sizes=(20.,30.),
                         young=1)

    if comm.size==1:
        assert system.__class__.__name__ == "FastSmoothContactSystem"
    else:
        assert system.__class__.__name__ == "SmoothContactSystem"

def test_choose_smooth_contactsystem(comm_self):
    """
    even on one processor, one should be able to force the usage of the
    smooth contact system. The occurence of jump instabilities make the babushka
    system difficult to use.

    """
    pass

def test_hardwall_as_string(comm, examplefile):
    fn, res, data = examplefile
    make_system(substrate="periodic",
                interaction="hardwall",
                surface=fn,
                physical_sizes=(1.,1.),
                young=1,
                communicator=comm)

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