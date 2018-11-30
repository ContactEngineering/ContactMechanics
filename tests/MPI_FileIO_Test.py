

import unittest
import numpy as np

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

from PyCo.Tools.MPI_FileIO import save_npy

@unittest.skipUnless(_withMPI,"requires mpi4py")
class test_MPI_npy(unittest.TestCase):
    @unittest.expectedFailure
    def test_FileSave_1D(self):
        comm = MPI.COMM_WORLD

        domain_resolution    = 128
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution // nprocs

        if rank == nprocs - 1:
            subdomain_slice = slice(rank * step, None)
            subdomain_location =rank * step
            subdomain_resolution = domain_resolution - rank * step
        else:
            subdomain_slice = slice(rank * step, (rank + 1) * step)
            subdomain_location = rank * step
            subdomain_resolution = step

        localdata = globaldata[subdomain_slice]

        save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,comm)

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data,globaldata)


    def test_FileSave_2D_slab_x(self):
        comm = MPI.COMM_WORLD

        domain_resolution    = (128,128)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution[0] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(rank * step, None),slice(None,None))
            subdomain_location =[rank * step,0]
            subdomain_resolution = [domain_resolution[0] - rank * step,domain_resolution[1]]
        else:
            subdomain_slice = (slice(rank * step, (rank + 1) * step),slice(None,None))
            subdomain_location = [rank * step,0]
            subdomain_resolution = [step,domain_resolution[1]]

        localdata = globaldata[subdomain_slice]


        save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,comm)

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data,globaldata)


    def test_FileSave_2D_slab_x(self):
        comm = MPI.COMM_WORLD

        domain_resolution    = (128,128)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution[0] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(rank * step, None),slice(None,None))
            subdomain_location =[rank * step,0]
            subdomain_resolution = [domain_resolution[0] - rank * step,domain_resolution[1]]
        else:
            subdomain_slice = (slice(rank * step, (rank + 1) * step),slice(None,None))
            subdomain_location = [rank * step,0]
            subdomain_resolution = [step,domain_resolution[1]]

        localdata = globaldata[subdomain_slice]


        save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,comm)

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data,globaldata)

    def test_FileSave_2D_slab_y(self):
        comm = MPI.COMM_WORLD

        domain_resolution = (128, 128)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution[1] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(None, None),slice(rank * step, None))
            subdomain_location = [0,rank * step]
            subdomain_resolution = [domain_resolution[0], domain_resolution[1] - rank * step]
        else:
            subdomain_slice = ( slice(None, None), slice(rank * step, (rank + 1) * step))
            subdomain_location = [0,rank * step]
            subdomain_resolution = [domain_resolution[1],step]

        localdata = globaldata[subdomain_slice]

        save_npy("test_Filesave_1D.npy", localdata, subdomain_location, domain_resolution, comm)

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data, globaldata)


    def test_FileLoad_2D_slab_x(self):
        pass