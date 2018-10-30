

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
    from PyCo.Tools.ParallelNumpy import ParallelNumpy

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace

from FFTEngine import NumpyFFTEngine
DEFAULTFFTENGINE = NumpyFFTEngine

@unittest.skipUnless(_withMPI,"requires mpi4py")
class test_ParallelNumpy(unittest.TestCase):

    def setUp(self):
        self.np = ParallelNumpy()
        self.comm = MPI.COMM_WORLD

    def test_array(self):
        nparr = np.array(((1,2,3),(4,5,6)))
        arr = self.np.array(((1,2,3),(4,5,6)))
        self.assertTrue(isinstance(arr, np.ndarray))
        self.assertTrue(np.array_equal(arr,nparr))

    def test_sum_scalar(self):
        res=self.np.sum(np.array(1))
        self.assertEqual(res, self.np.comm.Get_size())

    def test_sum_1D(self):
        arr=np.array((1,2.1,3))
        res = self.np.sum(arr)
        self.assertEqual(res, self.np.comm.Get_size() * 6.1)

    def test_sum_2D(self):
        arr=np.array(((1,2.1,3),
                     (4,5,6)))
        res = self.np.sum(arr)
        self.assertEqual(res, self.np.comm.Get_size() * 21.1)

    def test_max_2D(self):
        arr=np.reshape(np.array((-1,1,5,4,
                             4,5,4,5,
                             7,0,1,0)),(3,4))

        rank = self.comm.Get_rank()
        if self.comm.Get_size() >=4:
            if rank ==0 :   local_arr = arr[0:2,0:2]
            elif rank ==1 : local_arr = arr[0:2,2:]
            elif rank == 2 :local_arr = arr[2:,0:2]
            elif rank == 3 : local_arr = arr[2:,2:]
            else : local_arr = np.empty(0,dtype=arr.dtype)
        elif self.comm.Get_size() >=2 :
            if   rank ==0 :   local_arr = arr[0:2,:]
            elif rank ==1 : local_arr = arr[2:,:]
            else:           local_arr = np.empty(0, dtype=arr.dtype)
        else:
            local_arr = arr
        self.assertEqual(self.np.max(local_arr),7)


    def test_min(self):
        arr = np.reshape(np.array((-1, 1, 5, 4,
                                   4, 5, 4, 5,
                                   7, 0, 1, 0)), (3, 4))

        rank = self.comm.Get_rank()
        if self.comm.Get_size() >= 4:
            if rank == 0:
                local_arr = arr[0:2, 0:2]
            elif rank == 1:
                local_arr = arr[0:2, 2:]
            elif rank == 2:
                local_arr = arr[2:, 0:2]
            elif rank == 3:
                local_arr = arr[2:, 2:]
            else:
                local_arr = np.empty(0, dtype=arr.dtype)
        elif self.comm.Get_size() >= 2:
            if rank == 0:
                local_arr = arr[0:2, :]
            elif rank == 1:
                local_arr = arr[2:, :]
            else:
                local_arr = np.empty(0, dtype=arr.dtype)
        else:
            local_arr = arr
        self.assertEqual(self.np.min(local_arr), -1)

    def test_ones(self):
        print(np.ones is self.np.ones)
        print(np.ones)
        print(ParallelNumpy.ones)
        print(np.ones((10,10),dtype=bool))


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

        self.nx = 16
        self.ny = 32

        # equivalent Young's modulus
        self.E_s = 1.0
        # self.substrate = PeriodicFFTElasticHalfSpace(resolution=(32,32))

        self.comm = MPI.COMM_WORLD

        self.pnp = ParallelNumpy(comm=self.comm)

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
        Y, X = np.meshgrid(np.linspace(0, self.sy, self.ny + 1)[:-1], np.linspace(0, self.sx, self.nx + 1)[:-1])

        qx = 1 * np.pi * 2 / self.sx
        qy = 4 * np.pi * 2 / self.sy

        q = np.sqrt(qx ** 2 + qy ** 2)
        p = np.cos(qx * X + qy * Y)

        refdisp = - p / self.E_s * 2 / q
        # refpressure = PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=NumpyFFTEngine).evaluate_force(p)

        MPIsubstrates = [PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=engine)
                         for engine in [PFFTEngine]]

        for substrate in MPIsubstrates:
            computeddisp = substrate.evaluate_disp(p[substrate.subdomain_slice]*substrate.area_per_pt)

            np.testing.assert_allclose(computeddisp, refdisp[substrate.subdomain_slice], atol=1e-7, rtol=1e-2)

#    def test_k_force_maxq(self):
#        Y, X = np.meshgrid(np.linspace(0, self.sy, self.ny + 1)[:-1], np.linspace(0, self.sx, self.nx + 1)[:-1])
#
#        qx = 1 * np.pi * 2 / self.sx
#        qy = self.ny//2 * np.pi * 2 / self.sy
#
#        q = np.sqrt(qx ** 2 + qy ** 2)
#        h=1
#        disp = h*np.cos(qx * X + qy * Y)
#
#        ref_k_force= np.zeros((self.nx, self.ny//2+1))
#        ref_k_force[1,ny//2] = q * h *self.E_s /2


    def test_k_disp(self):
        pass

    def test_evaluate_elastic_energy(self):
        pass

    def test_evaluate(self):
        Y, X = np.meshgrid(np.linspace(0, self.sy, self.ny + 1)[:-1], np.linspace(0, self.sx, self.nx + 1)[:-1])

        disp = np.zeros((self.nx,self.ny))
        refForce = np.zeros((self.nx,self.ny))

        refEnergy = 0
        for qx,qy in zip((1 , 0,5,self.nx//2-1),
                         (4 , 4,0,self.ny//2-2)):
            qx = qx *np.pi * 2 / self.sx
            qy = qy* np.pi * 2 / self.sy

            q = np.sqrt(qx ** 2 + qy ** 2)
            h=1#q**(-0.8)
            disp += h *np.cos(qx * X + qy * Y)
            refForce += h *np.cos(qx * X + qy * Y) * self.E_s / 2 * q

            refEnergy+= self.E_s /8 * q * h**2

        # max possible Wavelengths at the edge
        for qx, qy in zip((self.nx//2 , self.nx//2   , 0            ),
                          (self.ny//2 ,   0          , self.ny//2 )):
            qx = qx *np.pi * 2 / self.sx
            qy = qy* np.pi * 2 / self.sy

            q = np.sqrt(qx ** 2 + qy ** 2)
            h=1#q**(-0.8)
            disp += h *np.cos(qx * X + qy * Y)
            refForce += h *np.cos(qx * X + qy * Y) * self.E_s / 2 * q

            refEnergy+= self.E_s /8 * q * h**2 * 2 # when the wavevector is equal to n//2 2pi/s (2 points per period),
            # the discretization is bad and it's not the energy of a cos wave anymore

        refEnergy *= self.sx * self.sy
        refForce *= -self.sx * self.sy / (self.nx * self.ny)

        MPIsubstrates = [PeriodicFFTElasticHalfSpace((self.nx, self.ny), self.E_s, (self.sx, self.sy), fftengine=engine)
                         for engine in [PFFTEngine]]
        for substrate in MPIsubstrates:
            computed_E_k_space = substrate.evaluate(disp[substrate.subdomain_slice],pot=True,forces=False)[0] # If force is not queried this computes the energy using kspace
            computed_E_realspace,computed_force = substrate.evaluate(disp[substrate.subdomain_slice],pot=True,forces=True)

            #print(self.pnp.sum(computed_E_k_space))
            #print(computed_E_k_space)
            #print(refEnergy)

            computed_E_k_space = self.pnp.sum(computed_E_k_space)
            computed_E_realspace = self.pnp.sum(computed_E_realspace)

            # Make an MPI-Reduce of the Energies !
            #print(substrate.evaluate_elastic_energy(refForce, disp))
            #print(0.5*np.vdot(refForce,disp))
            #print(substrate.evaluate_elastic_energy(substrate.evaluate_force(disp),disp))
            #print(computed_E_k_space)
            #print(computed_E_realspace)
            #print(refEnergy)

            self.assertAlmostEqual(computed_E_k_space,refEnergy)
            self.assertAlmostEqual(computed_E_realspace, refEnergy)
            np.testing.assert_allclose(computed_force, refForce[substrate.subdomain_slice], atol=1e-10, rtol=1e-10)

    def test_evaluate_elastic_energy_k_space_speed(self):
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

    def test_Hertzian_Pressure(self):
        """
        Apply Pressure Profile and look that the Deformation is hertzian
        Returns
        -------

        """
        pass

