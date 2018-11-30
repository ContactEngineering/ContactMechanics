

import unittest

from mpi4py import MPI
from PyCo.Topography.ParallelFromFile import read_npy
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace,PeriodicFFTElasticHalfSpace
from FFTEngine import PFFTEngine
from PyCo.System.Factory import SystemFactory
from PyCo.ContactMechanics.Interactions import HardWall

import numpy as np

class system_setup_workflow(unittest.TestCase):
    """

    represents the UseCase of creating System with MPI parallelization

    """
    def setUp(self):

        self.fn = "worflowtest.npy"

        self.res = (128,64)

        data  = np.random.random(self.res )
        data -= np.mean(data)

        np.save(self.fn,data)

    def test_workflow_superuser(self): # TODO: loop over all the cases, maybe with a pytest fixture  ?
        comm = MPI.COMM_WORLD

        interaction = HardWall()

        # Read metadata from the file and returns a UniformTopgraphy Object
        fileReader = read_npy(self.fn,headers_only=True)  # TODO: This mayBe of the lass Topography or a FileView Class

        assert fileReader.resolution == self.res

        # create a substrate according to the topography

        fftengine = PFFTEngine(domain_resolution = fileReader.resolution, comm = comm)
        Es = 1

        substrate = PeriodicFFTElasticHalfSpace(resolution=fileReader.resolution,size = fileReader.size,young = Es, fftengine=fftengine )

        top = fileReader.MPI_read(substrate)

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