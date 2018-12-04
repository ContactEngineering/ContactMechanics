

import unittest
import numpy as np
import os

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

from PyCo.Tools.MPIFileIO import save_npy, load_npy, MPIFileTypeError, MPIFileIncompatibleResolutionError, MPIFileView_npy

@unittest.skipUnless(_withMPI,"requires mpi4py")
class test_MPI_2D_npy(unittest.TestCase):

    def setUp(self):

        self.comm = MPI.COMM_WORLD

        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

        self.domain_resolution = (128,128)
        np.random.seed(2)
        self.globaldata = np.random.random(self.domain_resolution)

        if self.rank == 0 :
            np.save("test_FileLoad_2D.npy",self.globaldata)

        self.comm.barrier()

    def decomp_2D_slab_y(self):

        step = self.domain_resolution[1] // self.nprocs

        if self.rank == self.nprocs - 1:
            self.subdomain_slice = (slice(None, None), slice(self.rank * step, None))
            self.subdomain_location = [0, self.rank * step]
            self.subdomain_resolution = [self.domain_resolution[0], self.domain_resolution[1] - self.rank * step]
        else:
            self.subdomain_slice = (slice(None, None), slice(self.rank * step, (self.rank + 1) * step))
            self.subdomain_location = [0, self.rank * step]
            self.subdomain_resolution = [self.domain_resolution[1], step]

        self.localdata = self.globaldata[self.subdomain_slice]

    def decomp_2D_slab_x(self):
        step = self.domain_resolution[0] // self.nprocs

        if self.rank == self.nprocs - 1:
            self.subdomain_slice = (slice(self.rank * step, None), slice(None, None))
            self.subdomain_location = [self.rank * step, 0]
            self.subdomain_resolution = [self.domain_resolution[0] - self.rank * step, self.domain_resolution[1]]
        else:
            self.subdomain_slice = (slice(self.rank * step, (self.rank + 1) * step), slice(None, None))
            self.subdomain_location = [self.rank * step, 0]
            self.subdomain_resolution = [step, self.domain_resolution[1]]

        self.localdata = self.globaldata[self.subdomain_slice]

    def test_FileSave_2D(self):
        for decompfun in self.decomp_2D_slab_x,self.decomp_2D_slab_y:
            with self.subTest(decompfun = decompfun):
                decompfun()

                save_npy("test_Filesave_2D.npy",self.localdata,self.subdomain_location,self.domain_resolution,self.comm)
                loaded_data = np.load("test_Filesave_2D.npy")
                np.testing.assert_array_equal(loaded_data,self.globaldata)
        if self.rank == 0:
            os.remove("test_Filesave_2D.npy")


    def test_FileView_2D(self):
        for decompfun in self.decomp_2D_slab_x,self.decomp_2D_slab_y:
            with self.subTest(decompfun = decompfun):
                decompfun()


                #arr = np.load("test_FileLoad_2D.npy")
                #assert arr.shape == self.domain_resolution

                file = MPIFileView_npy("test_FileLoad_2D.npy", comm=self.comm)

                assert file.resolution == self.domain_resolution
                assert file.dtype == self.globaldata.dtype

                loaded_data = file.read(subdomain_resolution=self.subdomain_resolution,
                                       subdomain_location= self.subdomain_location)


                np.testing.assert_array_equal(loaded_data,self.localdata)

    def test_FileLoad_2D(self):
        for decompfun in self.decomp_2D_slab_x, self.decomp_2D_slab_y:
            with self.subTest(decompfun=decompfun):
                decompfun()

                # arr = np.load("test_FileLoad_2D.npy")
                # assert arr.shape == self.domain_resolution

                loaded_data = load_npy("test_FileLoad_2D.npy",
                                       subdomain_resolution=self.subdomain_resolution,
                                       subdomain_location=self.subdomain_location,
                                       domain_resolution=self.domain_resolution,
                                       comm=self.comm)

                np.testing.assert_array_equal(loaded_data, self.localdata)

                with self.assertRaises(MPIFileIncompatibleResolutionError):
                    load_npy("test_FileLoad_2D.npy",
                             subdomain_resolution=self.subdomain_resolution,
                             subdomain_location=self.subdomain_location,
                             domain_resolution=tuple([a + 1 for a in self.domain_resolution]),
                             comm=self.comm)

    def tearDown(self):
        if self.rank == 0 :
            os.remove("test_FileLoad_2D.npy")


class test_MPI_1D_npy(unittest.TestCase):
    @unittest.expectedFailure
    def test_FileSave_1D(self):
        self.comm = MPI.COMM_WORLD

        domain_resolution    = 128
        np.random.seed(2)
        self.globaldata = np.random.random(domain_resolution)

        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        step = domain_resolution // nprocs

        if rank == nprocs - 1:
            subdomain_slice = slice(rank * step, None)
            subdomain_location =rank * step
            subdomain_resolution = domain_resolution - rank * step
        else:
            subdomain_slice = slice(rank * step, (rank + 1) * step)
            subdomain_location = rank * step
            subdomain_resolution = step

        localdata = self.globaldata[subdomain_slice]

        save_npy("test_Filesave_1D.npy",localdata,subdomain_location,domain_resolution,self.comm)

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data,self.globaldata)