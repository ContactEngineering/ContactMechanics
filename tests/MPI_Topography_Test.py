

import unittest
import numpy as np

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

from PyCo.Topography import MPITopographyLoader

#from PyCo.Tools.MPI_FileIO import save_npy

class test_MPI_Topography(unittest.TestCase):
#    def setUp(self):
#        self.res= (128,64)
    def test_load_npy(self):
        return # tested in MPI_system_setup_test
        comm = MPI.COMM_WORLD

        domain_resolution = (128, 65)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution[0] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(rank * step, None), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [domain_resolution[0] - rank * step, domain_resolution[1]]
        else:
            subdomain_slice = (slice(rank * step, (rank + 1) * step), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [step, domain_resolution[1]]

        localdata = globaldata[subdomain_slice]

        if comm.Get_rank()==0:
            np.save("test_MPI_Topography.npy", globaldata)

        fileReader = MPITopographyLoader(self.fn, comm=comm)
        Top = UniformTopography(resolution=domain_resolution,
                                subdomain_location=subdomain_location,
                                subdomain_resolution=subdomain_resolution)

        Top.load_npy("test_Filesave_1D.npy")

        np.testing.assert_array_equal(Top.array(), localdata)

        # test rms_height Sq
        assert Top.rms_height() == np.sqrt()
        # test rms_height Rq





    @unittest.expectedFailure
    def test_load_npy_substrate(self):



        comm = MPI.COMM_WORLD

        domain_resolution = (128, 65)
        np.random.seed(2)
        globaldata = np.random.random(domain_resolution)

        nprocs = comm.Get_size()
        rank = comm.Get_rank()

        step = domain_resolution[0] // nprocs

        if rank == nprocs - 1:
            subdomain_slice = (slice(rank * step, None), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [domain_resolution[0] - rank * step, domain_resolution[1]]
        else:
            subdomain_slice = (slice(rank * step, (rank + 1) * step), slice(None, None))
            subdomain_location = [rank * step, 0]
            subdomain_resolution = [step, domain_resolution[1]]

        localdata = globaldata[subdomain_slice]

        np.save("test_MPI_Topography.npy", globaldata)

        Top = ParallelUniformTopography()

        loaded_data = np.load("test_Filesave_1D.npy")
        np.testing.assert_array_equal(loaded_data, globaldata)


    def test_save_npy(self):

        pass

    def test_rms_height(self):

        pass

    def test_rms_slope(self):

        pass

    def test_rms_curvature(self):
        pass


suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(test_MPI_Topography)])

if __name__ in  ['__main__','builtins']:
    print("Running unittest MPI_FileIO_Test")
    result = unittest.TextTestRunner().run(suite)