

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
from .PyCoTest import PyCoTestCase

from FFTEngine import NumpyFFTEngine
DEFAULTFFTENGINE = NumpyFFTEngine

@unittest.skipUnless(_withMPI,"requires mpi4py")
class test_ParallelNumpy(unittest.TestCase):

    def setUp(self):
        self.np = ParallelNumpy()
        self.comm = MPI.COMM_WORLD
        self.rank  = self.comm.Get_rank()
        self.MPIsize = self.comm.Get_size()
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
                             7,0,1,0.),dtype=float),(3,4))

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

    def test_max_min_empty(self):
        """
        Sometimes the input array is empty
        """
        if self.MPIsize >=2 :
            if self.rank==0:
                local_arr = np.array([], dtype=float)

            else :
                local_arr = np.array([1, 0, 4], dtype=float)
            self.assertEqual(self.np.max(local_arr), 4)
            self.assertEqual(self.np.min(local_arr), 0)

            if self.rank==0:
                local_arr = np.array([1, 0, 4], dtype=float)
            else :

                local_arr = np.array([], dtype=float)
            self.assertEqual(self.np.max(local_arr), 4)
            self.assertEqual(self.np.min(local_arr), 0)

        else :
            local_arr = np.array([],dtype = float)
            #self.assertTrue(np.isnan(self.np.max(local_arr)))
            #self.assertTrue(np.isnan(self.np.min(local_arr)))
            self.assertEqual(self.np.max(local_arr),-np.inf)
            self.assertEqual(self.np.min(local_arr),np.inf)



    def test_min(self):
        arr = np.reshape(np.array((-1, 1, 5, 4,
                                   4, 5, 4, 5,
                                   7, 0, 1, 0),dtype = float), (3, 4))

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



@unittest.skipUnless(_withMPI,"requires mpi4py")
class test_FFTElasticHalfSpace_weights(unittest.TestCase):
    """

    """
    def setUp(self):
        self.sx = 30.0
        self.sy = 1.0

        self.nx = 64
        self.ny = 33

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

class test_FFTElasticHalfSpace_compute(unittest.TestCase):
    def setUp(self):
        self.sx = 2  # 30.0
        self.sy = 1.0

        self.nx = 32
        self.ny = 64

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

            #print("{}: Local: E_kspace: {}, E_realspace: {}".format(substrate.fftengine.comm.Get_rank(),computed_E_k_space,computed_E_realspace))


            #print(computed_E_k_space)
            #print(refEnergy)

            computed_E_k_space = self.pnp.sum(np.array(computed_E_k_space,dtype = float))
            computed_E_realspace = self.pnp.sum(np.array(computed_E_realspace, dtype=float))

            #if substrate.fftengine.comm.Get_rank() == 0 :
            #    print(computed_E_k_space)
            #    print(computed_E_realspace)

            #print("{}: Global: E_kspace: {}, E_realspace: {}".format(substrate.fftengine.comm.Get_rank(),
                                                                     #computed_E_k_space, computed_E_realspace))

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
class test_FreeFFTElasticHalfSpace(PyCoTestCase):
    def setUp(self):
        self.E_s = 3
        self.fftengineList= [PFFTEngine]
        self.comm = MPI.COMM_WORLD
        self.pnp = ParallelNumpy(self.comm)

    def test_resolutions(self):
        nx,ny = 64,32
        sx,sy = 100,200

        for fftengine in self.fftengineList:
            with self.subTest(fftengine=fftengine):
                substrate=FreeFFTElasticHalfSpace((nx,ny),self.E_s,(sx,sy),fftengine=fftengine)
                self.assertEqual(substrate.resolution,(nx,ny))
                self.assertEqual(substrate.domain_resolution, (2*nx, 2*ny))

                self.assertEqual(self.pnp.sum(np.array(np.prod(substrate.subdomain_resolution))),4*nx*ny)

    def test_weights(self):
        def _compute_fourier_coeffs_serial_impl(self):
            """Compute the weights w relating fft(displacement) to fft(pressure):
               fft(u) = w*fft(p), Johnson, p. 54, and Hockney, p. 178

               This version is less is copied from matscipy, use if memory is a
               concern
            """
            # pylint: disable=invalid-name
            facts = np.zeros(tuple((res * 2 for res in self.resolution)))
            a = self.steps[0] * .5
            if self.dim == 1:
                pass
            else:
                b = self.steps[1] * .5
                x_s = np.arange(self.resolution[0] * 2)
                x_s = np.where(x_s <= self.resolution[0], x_s,
                               x_s - self.resolution[0] * 2) * self.steps[0]
                x_s.shape = (-1, 1)
                y_s = np.arange(self.resolution[1] * 2)
                y_s = np.where(y_s <= self.resolution[1], y_s,
                               y_s - self.resolution[1] * 2) * self.steps[1]
                y_s.shape = (1, -1)
                facts = 1 / (np.pi * self.young) * (
                        (x_s + a) * np.log(((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +
                                                                (x_s + a) * (x_s + a))) /
                                           ((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +
                                                                (x_s + a) * (x_s + a)))) +
                        (y_s + b) * np.log(((x_s + a) + np.sqrt((y_s + b) * (y_s + b) +
                                                                (x_s + a) * (x_s + a))) /
                                           ((x_s - a) + np.sqrt((y_s + b) * (y_s + b) +
                                                                (x_s - a) * (x_s - a)))) +
                        (x_s - a) * np.log(((y_s - b) + np.sqrt((y_s - b) * (y_s - b) +
                                                                (x_s - a) * (x_s - a))) /
                                           ((y_s + b) + np.sqrt((y_s + b) * (y_s + b) +
                                                                (x_s - a) * (x_s - a)))) +
                        (y_s - b) * np.log(((x_s - a) + np.sqrt((y_s - b) * (y_s - b) +
                                                                (x_s - a) * (x_s - a))) /
                                           ((x_s + a) + np.sqrt((y_s - b) * (y_s - b) +
                                                                (x_s + a) * (x_s + a)))))
            self.weights = np.fft.rfftn(facts)
            return self.weights, facts

        nx, ny = 64, 32
        sx, sy = 100, 200

        ref_weights, ref_facts = _compute_fourier_coeffs_serial_impl(
        FreeFFTElasticHalfSpace((nx, ny), self.E_s, (sx, sy), fftengine=NumpyFFTEngine))

        for fftengine in self.fftengineList:
            with self.subTest(fftengine=fftengine):
                substrate=FreeFFTElasticHalfSpace((nx,ny),self.E_s,(sx,sy),fftengine=fftengine)
                local_weights,local_facts = substrate._compute_fourier_coeffs()
                np.testing.assert_allclose(local_weights,ref_weights[substrate.fourier_slice],1e-12)
                np.testing.assert_allclose(local_facts,ref_facts[substrate.subdomain_slice],1e-12)

    def test_evaluate_disp_uniform_pressure(self):
        nx, ny = 64, 32
        sx, sy = 100, 200

        forces = np.zeros((2 * nx, 2 * ny))
        x = (np.arange(2 * nx) - (nx - 1) / 2) * sx / nx
        x.shape = (-1, 1)
        y = (np.arange(2 * ny) - (ny - 1) / 2) * sy / ny
        y.shape = (1, -1)

        forces[1:nx - 1, 1:ny - 1] = -np.ones((nx - 2, ny - 2)) * (sx * sy) / (nx * ny)
        a = (nx - 2) / 2 * sx / nx
        b = (ny - 2) / 2 * sy / ny
        refdisp = 1/(np.pi*self.E_s) * (
                (x+a)*np.log(((y+b)+np.sqrt((y+b)*(y+b) +
                                                (x+a)*(x+a))) /
                               ((y-b)+np.sqrt((y-b)*(y-b) +
                                                (x+a)*(x+a)))) +
                (y+b)*np.log(((x+a)+np.sqrt((y+b)*(y+b) +
                                                (x+a)*(x+a))) /
                               ((x-a)+np.sqrt((y+b)*(y+b) +
                                                (x-a)*(x-a)))) +
                (x-a)*np.log(((y-b)+np.sqrt((y-b)*(y-b) +
                                                (x-a)*(x-a))) /
                               ((y+b)+np.sqrt((y+b)*(y+b) +
                                                (x-a)*(x-a)))) +
                (y-b)*np.log(((x-a)+np.sqrt((y-b)*(y-b) +
                                                (x-a)*(x-a))) /
                               ((x+a)+np.sqrt((y-b)*(y-b) +
                                                (x+a)*(x+a)))))

        for fftengine in self.fftengineList:
            with self.subTest(fftengine=fftengine):
                substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s, (sx, sy), fftengine=PFFTEngine)

                if self.comm.Get_size() > 1:
                    with self.assertRaises(FreeFFTElasticHalfSpace.Error):
                        substrate.evaluate_disp(forces)
                    with self.assertRaises(FreeFFTElasticHalfSpace.Error):
                        substrate.evaluate_disp(forces[nx, ny])

                #print(forces[substrate.subdomain_slice])
                computed_disp = substrate.evaluate_disp(forces[substrate.subdomain_slice])
                #print(computed_disp)
                # make the comparison only on the nonpadded domain
                s_c = tuple([slice(1, max(0, min(substrate.resolution[i]-1 - substrate.subdomain_location[i], substrate.subdomain_resolution[i])))
                         for i in range(substrate.dim)])

                s_refdisp = tuple([slice(s_c[i].start + substrate.subdomain_location[i], s_c[i].stop + substrate.subdomain_location[i]) for i in range(substrate.dim)])
                #print(s_c)
                #print(s_refdisp)
                np.testing.assert_allclose(computed_disp[s_c],refdisp[s_refdisp])


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

if __name__ == '__main__':
    unittest.main()