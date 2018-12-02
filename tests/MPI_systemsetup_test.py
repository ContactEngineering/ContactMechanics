

import unittest

from mpi4py import MPI
from PyCo.Topography.ParallelFromFile import read_npy
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace,PeriodicFFTElasticHalfSpace
from FFTEngine import PFFTEngine
from PyCo.System.Factory import SystemFactory
from PyCo.ContactMechanics.Interactions import HardWall
from PyCo.Topography import MPITopographyLoader

import numpy as np

class system_setup_workflow(unittest.TestCase):
    """

    represents the UseCase of creating System with MPI parallelization

    """
    def setUp(self):
        self.fn = "worflowtest.npy"
        self.res = (128,64)
        np.random.seed(1)
        self.data  = np.random.random(self.res )
        self.data -= np.mean(self.data)

        np.save(self.fn,self.data)

    def test_setup_from_topoFile_superuser(self): # TODO: loop over all the cases, maybe with a pytest fixture  ?
        comm = MPI.COMM_WORLD

        for HS in [PeriodicFFTElasticHalfSpace,FreeFFTElasticHalfSpace]:
            with self.subTest(HS=HS):
                interaction = HardWall()

                # Read metadata from the file and returns a UniformTopgraphy Object
                fileReader = MPITopographyLoader(self.fn,comm=comm)

                assert fileReader.resolution == self.res

                # create a substrate according to the topography

                fftengine = PFFTEngine(domain_resolution = fileReader.resolution, comm = comm)
                Es = 1
                if fileReader.size is not None:
                    substrate = HS(resolution=fileReader.resolution,size = fileReader.size,young = Es, fftengine=fftengine )
                else:
                    substrate = HS(resolution=fileReader.resolution,young = Es, fftengine=fftengine )

                top = fileReader.getTopography(substrate)

                assert top.resolution == substrate.resolution
                assert top.subdomain_resolution == substrate.subdomain_resolution \
                       or top.subdomain_resolution == (0,0) # for FreeFFTElHS
                assert top.subdomain_location == substrate.subdomain_location

                np.testing.assert_array_equal(top.array(),self.data[top.subdomain_slice])

                system = SystemFactory(substrate,interaction,top)

        # make some tests on the system

    @unittest.expectedFailure
    def test_workflow_casualuser(self):
        """
        longtermgoal for confortable and secure use
        Returns
        -------

        """
        # Maybe it will be another Function or class

        substrate =  PeriodicFFTElasticHalfSpace
        interaction = HardWall

        system = SystemFactory(substrate,interaction,self.fn)