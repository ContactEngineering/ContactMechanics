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
import numpy as np

try:
    from mpi4py import MPI

    _withMPI = True
except ImportError:
    print("No MPI")
    _withMPI = False

if _withMPI:
    from FFTEngine import PFFTEngine
    from FFTEngine.helpers import gather
from FFTEngine import NumpyFFTEngine

DEFAULTFFTENGINE = NumpyFFTEngine

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace

from NuMPI.Tools import ParallelNumpy

comm = MPI.COMM_WORLD
pnp = ParallelNumpy(comm=comm)
fftengineList = [PFFTEngine]

def test_weights_gather(fftengineclass, nx=64, ny=33):
    """

    """

    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 1.0
    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fftengine=fftengineclass((nx, ny), comm), pnp=pnp)
    fourres = (substrate.domain_resolution[0], substrate.domain_resolution[1] // 2 + 1)
    weights = gather(substrate.weights, substrate.fourier_slice, fourres, comm, root=0)
    iweights = gather(substrate.iweights, substrate.fourier_slice, fourres, comm, root=0)
    if comm.Get_rank() == 0:
        reference = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                fftengine=DEFAULTFFTENGINE((nx, ny), comm=comm))
        np.testing.assert_allclose(reference.weights, weights, rtol=0, atol=1e-16,
                                   err_msg="weights are different after gather")
        np.testing.assert_allclose(reference.iweights, iweights, rtol=0, atol=1e-16,
                                   err_msg="iweights are different after gather")


def test_weights(fftengineclass, nx=64, ny=33): # TODO: merge the serial test of the weights into this
    sx = 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 1.0

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fftengine=fftengineclass((nx, ny), comm), pnp=pnp)
    reference = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fftengine=DEFAULTFFTENGINE((nx, ny), comm=comm))
    np.testing.assert_allclose(reference.weights[substrate.fourier_slice], substrate.weights, rtol=0, atol=1e-16,
                               err_msg="weights are different")
    np.testing.assert_allclose(reference.iweights[substrate.fourier_slice], substrate.iweights, rtol=0, atol=1e-16,
                               err_msg="iweights are different")


def test_sineWave_disp(fftengineclass, nx=64, ny=32):
    sx = 2.45  # 30.0
    sy = 1.0

    # equivalent Young's modulus
    E_s = 1.0

    for k in [(1,0), (0,1),(1, 2), (nx//2,0),(1,ny//2),(0,2),(nx//2,ny//2),(0,ny//2)]:
        #print("testing wavevector ({}* np.pi * 2 / sx, {}* np.pi * 2 / sy) ".format(*k))
        qx = k[0] * np.pi * 2 / sx
        qy = k[1] * np.pi * 2 / sy
        q = np.sqrt(qx ** 2 + qy ** 2)

        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])
        disp = np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)

        refpressure = - disp * E_s / 2 * q
        #np.testing.assert_allclose(refpressure,
        #    PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
        #    fftengine=NumpyFFTEngine((nx,ny))).evaluate_force(disp) / (sx*sy / (nx*ny)))

        substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                     fftengine=fftengineclass((nx, ny), comm), pnp=pnp)

        kpressure = substrate.evaluate_k_force(disp[substrate.subdomain_slice]) / substrate.area_per_pt / (nx*ny)
        expected_k_disp = np.zeros((nx, ny//2 + 1), dtype=complex)
        expected_k_disp[k[0], k[1]] += .5 - .5j

        # add the symetrics
        if k[1] == 0:
            expected_k_disp[-k[0], 0] += .5 + .5j
        if k[1] == ny//2 and ny%2 == 0 :
            expected_k_disp[-k[0], k[1]] += .5 + .5j

        np.testing.assert_allclose(substrate.fftengine.rfftn(disp[substrate.subdomain_slice]) / (nx*ny),
                                   expected_k_disp[substrate.fourier_slice], rtol=1e-7, atol=1e-10)

        expected_k_pressure = - E_s / 2 * q * expected_k_disp
        np.testing.assert_allclose(kpressure, expected_k_pressure[substrate.fourier_slice], rtol=1e-7, atol=1e-10)

        computedpressure = substrate.evaluate_force(disp[substrate.subdomain_slice]) / substrate.area_per_pt
        np.testing.assert_allclose(computedpressure, refpressure[substrate.subdomain_slice], atol=1e-10, rtol=1e-7)

        computedenergy_kspace = substrate.evaluate(disp[substrate.subdomain_slice], pot=True, forces=False)[0]
        computedenergy = substrate.evaluate(disp[substrate.subdomain_slice], pot=True, forces=True)[0]
        refenergy = E_s / 8 * 2 * q * sx * sy

        #print(substrate.domain_resolution[-1] % 2)
        #print(substrate.fourier_resolution)
        #print(substrate.fourier_location[-1] + substrate.fourier_resolution[-1] - 1)
        #print(substrate.domain_resolution[-1] // 2 )
        #print(computedenergy)
        #print(computedenergy_kspace)
        #print(refenergy)
        np.testing.assert_allclose(computedenergy, refenergy, rtol=1e-10,
                                   err_msg="wavevektor {} for domain_resolution {}, subdomain resolution {}, fourier_resolution {}".format(k, substrate.domain_resolution, substrate.subdomain_resolution, substrate.fourier_resolution))
        np.testing.assert_allclose(computedenergy_kspace, refenergy, rtol=1e-10,
                                   err_msg="wavevektor {} for domain_resolution {}, subdomain resolution {}, fourier_resolution {}".format(k, substrate.domain_resolution, substrate.subdomain_resolution, substrate.fourier_resolution))


def test_sineWave_disp_rotation_invariance(fftengineclass, nx=64, ny=32):
    sx = 3.  # 30.0
    sy = 3.

    # equivalent Young's modulus
    E_s = 1.0

    computedenergies=[]
    computedenergies_kspace=[]
    for k in [(min(nx,ny)//2, 0), (0, min(nx,ny)//2)]:
        qx = k[0] * np.pi * 2 / sx
        qy = k[1] * np.pi * 2 / sy
        q = np.sqrt(qx ** 2 + qy ** 2)

        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])
        disp = np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)  # At the Nyquist frequency for even nuimber of points, the energy computation can only be exact for this point

        refpressure = - disp * E_s / 2 * q
        #np.testing.assert_allclose(refpressure,
        #    PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
        #    fftengine=NumpyFFTEngine((nx,ny))).evaluate_force(disp) / (sx*sy / (nx*ny)))

        substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                                     fftengine=fftengineclass((nx, ny), comm), pnp=pnp)

        computedenergies_kspace +=[substrate.evaluate(disp[substrate.subdomain_slice], pot=True, forces=False)[0]]
        computedenergies += [substrate.evaluate(disp[substrate.subdomain_slice], pot=True, forces=True)[0]]

        refenergy = E_s / 8 * 1 * q * sx * sy

    #np.testing.assert_allclose(computedpressures[0],computedpressures[1].T)
    np.testing.assert_allclose(*computedenergies, rtol=1e-10)
    np.testing.assert_allclose(*computedenergies_kspace, rtol=1e-10)


def test_sineWave_force(fftengineclass,nx=64, ny=32):
    sx = 2  # 30.0
    sy = 1.0

    # equivalent Young's modulus
    E_s = 1.0

    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])

    qx = 1 * np.pi * 2 / sx
    qy = 4 * np.pi * 2 / sy

    q = np.sqrt(qx ** 2 + qy ** 2)
    p = np.cos(qx * X + qy * Y)

    refdisp = - p / E_s * 2 / q
    # refpressure = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy), fftengine=NumpyFFTEngine).evaluate_force(p)

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fftengine=fftengineclass((nx, ny), comm), pnp=pnp)
    computeddisp = substrate.evaluate_disp(p[substrate.subdomain_slice] * substrate.area_per_pt)
    np.testing.assert_allclose(computeddisp, refdisp[substrate.subdomain_slice], atol=1e-7, rtol=1e-10)

    # computedenergy = substrate.evaluate(p[substrate.subdomain_slice]*substrate.area_per_pt)

    # refenergy = sx * sy/(2 * q * E_s) * 1
    # np.testing.assert_allclose(computedenergy,refenergy,rtol = 1e-4)

#    def test_k_force_maxq(self):
#        Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])
#
#        qx = 1 * np.pi * 2 / sx
#        qy = ny//2 * np.pi * 2 / sy
#
#        q = np.sqrt(qx ** 2 + qy ** 2)
#        h=1
#        disp = h*np.cos(qx * X + qy * Y)
#
#        ref_k_force= np.zeros((nx, ny//2+1))
#        ref_k_force[1,ny//2] = q * h *E_s /2


def test_multipleSineWaves_evaluate(fftengineclass,nx=64, ny=32):
    sx = 2  # 30.0
    sy = 1.0
    # equivalent Young's modulus
    E_s = 1.0

    Y, X = np.meshgrid(np.linspace(0, sy, ny + 1)[:-1], np.linspace(0, sx, nx + 1)[:-1])

    disp = np.zeros((nx, ny))
    refForce = np.zeros((nx, ny))

    refEnergy = 0
    for qx, qy in zip((1, 0, 5, nx // 2 - 1),
                      (4, 4, 0, ny // 2 - 2)):
        qx = qx * np.pi * 2 / sx
        qy = qy * np.pi * 2 / sy

        q = np.sqrt(qx ** 2 + qy ** 2)
        h = 1  # q**(-0.8)
        disp += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y))
        refForce += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)) * E_s / 2 * q
        refEnergy += E_s / 8 * q * 2 * h ** 2
        # * 2 because the amplitude of cos(x) + sin(x) is sqrt(2)

    # max possible Wavelengths at the edge

    for qx, qy in zip((nx // 2, nx // 2, 0),
                      (ny // 2, 0, ny // 2)):
        qx = qx * np.pi * 2 / sx
        qy = qy * np.pi * 2 / sy

        q = np.sqrt(qx ** 2 + qy ** 2)
        h = 1  # q**(-0.8)
        disp += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y))
        refForce += h * (np.cos(qx * X + qy * Y) + np.sin(qx * X + qy * Y)) * E_s / 2 * q

        refEnergy += E_s / 8 * q * h ** 2 * 2
        # * 2 because the amplitude of cos(x) + sin(x) is sqrt(2)

    refEnergy *= sx * sy
    refForce *= -sx * sy / (nx * ny)

    substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sy),
                                            fftengine=fftengineclass((nx, ny), comm), pnp=pnp)
    computed_E_k_space = substrate.evaluate(disp[substrate.subdomain_slice], pot=True, forces=False)[0]
    # If force is not queried this computes the energy using kspace
    computed_E_realspace, computed_force = substrate.evaluate(disp[substrate.subdomain_slice], pot=True,
                                                              forces=True)

    # print("{}: Local: E_kspace: {}, E_realspace: {}".format(substrate.fftengine.comm.Get_rank(),computed_E_k_space,computed_E_realspace))
    # print(computed_E_k_space)
    # print(refEnergy)

    # if substrate.fftengine.comm.Get_rank() == 0 :
    #    print(computed_E_k_space)
    #    print(computed_E_realspace)

    # print("{}: Global: E_kspace: {}, E_realspace: {}".format(substrate.fftengine.comm.Get_rank(),
    # computed_E_k_space, computed_E_realspace))

    # Make an MPI-Reduce of the Energies !
    # print(substrate.evaluate_elastic_energy(refForce, disp))
    # print(0.5*np.vdot(refForce,disp))
    # print(substrate.evaluate_elastic_energy(substrate.evaluate_force(disp),disp))
    # print(computed_E_k_space)
    # print(computed_E_realspace)
    # print(refEnergy)

    np.testing.assert_almost_equal(computed_E_k_space, refEnergy)
    np.testing.assert_almost_equal(computed_E_realspace, refEnergy)
    np.testing.assert_allclose(computed_force, refForce[substrate.subdomain_slice], atol=1e-7, rtol=1e-10)


if __name__ in ['__main__', 'builtins']:
    basenpoints = comm.Get_size() * 4 # Base number of points in order to avoid empty subdomains when using a lot of processors 
    for fftengineclass in fftengineList:
        print("Testing rotational invartiance of energy")
        test_sineWave_disp_rotation_invariance(fftengineclass, basenpoints+8, basenpoints+8)
        test_sineWave_disp_rotation_invariance(fftengineclass, basenpoints+17, basenpoints+128)
        test_sineWave_disp_rotation_invariance(fftengineclass, basenpoints+16, basenpoints+128)
        print("rotation invariance ok")

        test_sineWave_disp(fftengineclass, nx=basenpoints+8, ny=basenpoints+15)
        test_sineWave_disp(fftengineclass, nx=basenpoints+8, ny=basenpoints+4)
        test_sineWave_disp(fftengineclass, nx=basenpoints+9, ny=basenpoints+4)
        test_sineWave_disp(fftengineclass, nx=basenpoints+113, ny=basenpoints+765)

        print("Testing Periodic FFTElastic Halfspace weights")
        test_weights_gather(fftengineclass, nx=basenpoints+64, ny=basenpoints+33)
        test_weights(fftengineclass, nx=basenpoints+64, ny=basenpoints+33)
        test_weights(fftengineclass, nx=basenpoints+65, ny=basenpoints+33)
        test_weights(fftengineclass, nx=basenpoints+64, ny=basenpoints+32)

        print("Testing Free FFTElastic Halfspace computation")
        for res in [(basenpoints+64, basenpoints+32), (basenpoints+65, basenpoints+33)]:
            print("testing resolution {}".format(res))
            test_sineWave_disp(fftengineclass, *res)
            test_sineWave_force(fftengineclass, *res)
            test_multipleSineWaves_evaluate(fftengineclass, *res)
