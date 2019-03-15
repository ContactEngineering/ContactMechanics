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


import unittest

from mpi4py import MPI
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace,PeriodicFFTElasticHalfSpace
from FFTEngine import PFFTEngine
from PyCo.System.Factory import make_system
from PyCo.ContactMechanics.Interactions import HardWall
from PyCo.Topography import MPITopographyLoader
from PyCo.Topography.ParallelFromFile import TopographyLoaderNPY
import numpy as np

class MPI_TopographyLoading_Test(unittest.TestCase):
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

    def test_LoadTopoFromFile(self): # TODO: loop over all the cases, maybe with a pytest fixture  ?
        comm = MPI.COMM_WORLD

        for HS in [PeriodicFFTElasticHalfSpace,FreeFFTElasticHalfSpace]:
            with self.subTest(HS=HS):
                for loader in [MPITopographyLoader, TopographyLoaderNPY]: #TODO: these implementations are redundant, only one of them will persist
                    with self.subTest(loader=loader):
                        interaction = HardWall()

                        # Read metadata from the file and returns a UniformTopgraphy Object
                        fileReader = MPITopographyLoader(self.fn, comm=comm)

                        assert fileReader.resolution == self.res

                        # create a substrate according to the topography

                        fftengine = PFFTEngine(domain_resolution = fileReader.resolution, comm = comm)
                        Es = 1
                        if fileReader.size is not None:
                            substrate = HS(resolution=fileReader.resolution, size=fileReader.size, young=Es, fftengine=fftengine )
                        else:
                            substrate = HS(resolution=fileReader.resolution, young = Es, fftengine=fftengine )

                        top = fileReader.topography(substrate)

                        assert top.resolution == substrate.resolution
                        assert top.subdomain_resolution == substrate.subdomain_resolution \
                               or top.subdomain_resolution == (0,0) # for FreeFFTElHS
                        assert top.subdomain_location == substrate.subdomain_location

                        np.testing.assert_array_equal(top.heights(),self.data[top.subdomain_slice])

                        # test that the slicing is what is expected

                        fulldomain_field = np.arange(np.prod(substrate.domain_resolution)).reshape(substrate.domain_resolution)

                        np.testing.assert_array_equal(fulldomain_field[top.subdomain_slice],fulldomain_field[tuple([slice(substrate.subdomain_location[i],substrate.subdomain_location[i]+max(0,min(substrate.resolution[i] - substrate.subdomain_location[i],substrate.subdomain_resolution[i]))) for i in range(substrate.dim)])])

                        # Test Computation of the rms_height
                        # Sq
                        assert top.rms_height(kind="Sq") == np.sqrt(np.mean((self.data - np.mean(self.data))**2))
                        #Rq
                        assert top.rms_height(kind="Rq") == np.sqrt(np.mean((self.data - np.mean(self.data,axis = 0))**2))

                        system = make_system(substrate, interaction, top)

        # make some tests on the system

    @unittest.expectedFailure
    def test_make_system_from_file(self):
        """
        longtermgoal for confortable and secure use
        Returns
        -------

        """
        # Maybe it will be another Function or class

        substrate =  PeriodicFFTElasticHalfSpace
        interaction = HardWall

        system = make_system(substrate, interaction, self.fn)


suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(MPI_TopographyLoading_Test)])
if __name__ in  ['__main__','builtins']:
    print("Running unittest MPI_FileIO_Test")
    result = unittest.TextTestRunner().run(suite)