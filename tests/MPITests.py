

import unittest
import numpy as np

try :
    from mpi4py import MPI
    _withMPI=True

except ImportError:
    print("No MPI")
    _withMPI =False

if _withMPI:
    from FFTEngine import PFFTEngine
    from FFTEngine.helpers import gather

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace

from FFTEngine import NumpyFFTEngine
DEFAULTFFTENGINE = NumpyFFTEngine

@unittest.skipUnless(_withMPI,"requires mpi4py")
class Parallel_FFTElasticHalfSpace_weights(unittest.TestCase):
    """

    """

    def setUp(self):
        self.sx = 30.0
        self.sy = 1.0

        self.nx = 16
        self.ny = 9

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

class Parallel_FFTElasticHalfSpace_compute(unittest.TestCase):
    def setUp(self):
        self.sx = 2  # 30.0
        self.sy = 1.0

        self.nx = 32
        self.ny = 64

        # equivalent Young's modulus
        self.E_s = 1.0
        # self.substrate = PeriodicFFTElasticHalfSpace(resolution=(32,32))

        self.comm = MPI.COMM_WORLD
    def test_disp_sineWave(self):
        Y,X = np.meshgrid(np.linspace(0,self.sy,self.ny+1)[:-1],np.linspace(0,self.sx,self.nx+1)[:-1])

        qx = 1 *np.pi * 2 / self.sx
        qy = 4 *np.pi * 2 / self.sy

        q = np.sqrt(qx**2 + qy**2)
        disp= np.cos(qx * X +qy* Y)

        refpressure = - disp * self.E_s / 2 * q
        #refpressure = PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=NumpyFFTEngine).evaluate_force(disp)

        MPIsubstrates = [PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=engine)
                         for engine in [PFFTEngine]]

        for substrate in MPIsubstrates:
            computedpressure = substrate.evaluate_force(disp[substrate.subdomain_slice]) / substrate.area_per_pt

            np.testing.assert_allclose(computedpressure, refpressure[substrate.subdomain_slice],atol = 1e-7, rtol = 1e-2)

    def test_force_sineWave(self):
        pass

    def test_k_force(self):
        pass
    def test_k_disp(self):
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

