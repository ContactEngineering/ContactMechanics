

import unittest
import numpy as np

try :
    from mpi4py import MPI
    _withMPI=True
    from FFTEngine import PFFTEngine
    from FFTEngine.helpers import gather
except ImportError:
    print("No MPI")
    _withMPI =False

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace

from FFTEngine import NumpyFFTEngine
DEFAULTFFTENGINE = NumpyFFTEngine

@unittest.skipUnless(_withMPI,"requires mpi4py")
class Parallel_FFTElasticHalfSpace(unittest.TestCase):
    """

    """

    def setUp(self):
        self.sx = 1.0#30.0
        self.sy = 1.0

        self.nx = 8
        self.ny = 8

        # equivalent Young's modulus
        self.E_s = 1.0
        #self.substrate = PeriodicFFTElasticHalfSpace(resolution=(32,32))

        self.comm = MPI.COMM_WORLD

    def test_weights_gather(self):
        """
        Compare weights with non parralel version

        Returns
        -------

        """

        MPIsubstrates = [PeriodicFFTElasticHalfSpace((self.nx,self.ny),self.E_s,(self.sx, self.sy),fftengine= engine) for engine in [PFFTEngine]]
        for substrate in MPIsubstrates:
            fourres = (substrate.domain_resolution[0], substrate.domain_resolution[1] //2 +1 )
            weights = gather(substrate.weights,substrate.fourier_slice,fourres,self.comm,root=0)
            iweights = gather(substrate.iweights,substrate.fourier_slice,fourres,self.comm,root=0)
            if self.comm.Get_rank() == 0:
                reference = PeriodicFFTElasticHalfSpace((self.nx,self.ny),self.E_s,(self.sx, self.sy),fftengine=DEFAULTFFTENGINE)
                np.testing.assert_allclose(reference.weights,weights,rtol=0, atol=1e-16, err_msg="weights are different after gather")
                np.testing.assert_allclose(reference.iweights, iweights, rtol=0, atol=1e-16,err_msg="iweights are different after gather")

    def test_weights(self):
        MPIsubstrates = [PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=engine)
                         for engine in [PFFTEngine]]
        for substrate in MPIsubstrates:

            reference = PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy),
                                                        fftengine=DEFAULTFFTENGINE)
            np.testing.assert_allclose(reference.weights[substrate.fourier_slice], substrate.weights, rtol=0, atol=1e-16,err_msg="weights are different")
            np.testing.assert_allclose(reference.iweights[substrate.fourier_slice], substrate.iweights, rtol=0, atol=1e-16, err_msg="iweights are different")

    def test_sineFunction(self):
        pass

@unittest.skipUnless(_withMPI,"requires mpi4py")
class Parallel_FreeFFTElasticHalfSpace(unittest.TestCase):

    def test_Parabolic_Displacement(self):
        """

        Apply hertzian Displacement Field and look that the hertzian pressure-profile comes out
        Returns
        -------

        """
        pass

    def test_Hertzian_Pressurez(self):
        """
        Apply Pressure Profile and look that the Deformation is hertzian
        Returns
        -------

        """
        pass

