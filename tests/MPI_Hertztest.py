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
try:
    import unittest
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import make_sphere
    from PyCo.System import NonSmoothContactSystem

    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions import Hertz as Hz

   # (radius_and_pressure,
   #  surface_displacements,
   #  surface_stress,
   #  penetration,
   #  normal_load)

    from mpi4py import MPI
    from FFTEngine import PFFTEngine
    from MPITools.Tools.ParallelNumpy import ParallelNumpy

except ImportError as err:
    import sys
    print(err)
    sys.exit(-1)

DEBUG=False
# -----------------------------------------------------------------------------
class HertzTest(unittest.TestCase):
    def setUp(self):
        # sphere radius:
        self.r_s = 20.0
        # contact radius
        self.r_c = .2
        # peak pressure
        self.p_0 = 2.5
        # equivalent Young's modulus
        self.E_s = 102.
        self.comm = MPI.COMM_WORLD
        self.pnp = ParallelNumpy(self.comm)

    def test_elastic_solution(self):
        r = np.linspace(0, self.r_s, 6)/self.r_c
        u = Hz.surface_displacements(r) / (self.p_0/self.E_s*self.r_c)
        sig = Hz.surface_stress(r)[0]/self.p_0

    def test_constrained_conjugate_gradients(self):
        for nx, ny in [(512, 512)]:#, (256, 255), (255, 256)]:
            for disp0, normal_force in [(0.1, None), (0, 15.0)]:
                sx = 5.0

                substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s,
                                                    (sx, sx), fftengine=PFFTEngine((2*nx, 2*ny),comm=self.comm),pnp=self.pnp)

                interaction = HardWall()
                surface = make_sphere(self.r_s, (nx, ny), (sx, sx),
                                 subdomain_resolution= substrate.topography_subdomain_resolution,
                                 subdomain_location=substrate.topography_subdomain_location,
                                 pnp = substrate.pnp)
                system = NonSmoothContactSystem(substrate, interaction, surface)

                result = system.minimize_proxy(offset=disp0,
                                               external_force=normal_force,
                                               kind="ref")
                disp = result.x
                forces = -result.jac
                converged = result.success
                self.assertTrue(converged)

                comp_normal_force = -self.pnp.sum(forces)
                if normal_force is not None:

                    self.assertAlmostEqual(normal_force, comp_normal_force,
                        msg="Convergence Problem: resultant normal Force doesn't match imposed normal force")
                    self.assertAlmostEqual(system.compute_normal_force(),normal_force,msg = "computed normal force doesn't match imposed force" )
                    # assert the disp is OK with analytical Solution

                    #print("penetration: computed: {}"
                    #      "               Hertz : {}".format(result.offset,penetration(normal_force,self.r_s,self.E_s)))
                    np.testing.assert_allclose(result.offset,Hz.penetration(normal_force,self.r_s,self.E_s),rtol=1e-2,
                        err_msg="computed offset doesn't match with hertz theory for imposed Load {}".format(normal_force))
                    Eel_ref = Hz.elastic_energy(Hz.penetration(normal_force,self.r_s,self.E_s), self.r_s, self.E_s)

                elif disp0 is not None:
                    Eel_ref = Hz.elastic_energy(disp0, self.r_s, self.E_s)
                    self.assertAlmostEqual(disp0, result.offset,msg="Convergence Problem: computed penetration doesn't match imposed penetration")
                    np.testing.assert_allclose(comp_normal_force,Hz.normal_load(disp0,self.r_s,self.E_s),rtol=1e-2,
                        err_msg="computed normal force doesn't match with hertz theory for imposed Penetration {}".format(disp0))
                    np.testing.assert_allclose(system.compute_normal_force(), Hz.normal_load(disp0,self.r_s,self.E_s),rtol=1e-2,
                                           err_msg="computed normal force doesn't match with hertz theory for imposed Penetration {}".format(disp0))


                a, p0 = Hz.radius_and_pressure(comp_normal_force, self.r_s,self.E_s)

                np.testing.assert_allclose(system.compute_contact_area(),np.pi*a**2,rtol=1e-1,err_msg="Computed area doesn't match Hertz Theory")

                Eel_computed_kspace = system.substrate.evaluate(disp,pot=True,forces=False)[0]
                Eel_computed_rspace = system.substrate.evaluate(disp,pot=True,forces=True)[0]


                np.testing.assert_allclose(Eel_computed_kspace,Eel_ref,rtol=1e-2)
                np.testing.assert_allclose(Eel_computed_rspace,Eel_ref,rtol=1e-2)

                p_numerical = -forces * (nx * ny / (sx * sx))
                p_analytical = np.zeros_like(p_numerical)

                if DEBUG:
                    for i in range(substrate.fftengine.comm.Get_size()):
                        substrate.fftengine.comm.barrier()
                        if substrate.fftengine.comm.Get_rank() == i:
                            print(i)
                            print("subdom_res:  %s" % substrate.subdomain_resolution.__repr__())
                            print("shape p_numerical: %s" % p_numerical.shape.__repr__())
                        else:
                            continue

                if np.prod(substrate.subdomain_resolution) > 0:
                    x = ((np.arange(nx) - nx / 2) * sx / nx).reshape(-1, 1)[substrate.subdomain_slice[0], :]
                    y = ((np.arange(ny) - ny / 2) * sx / ny).reshape(1, -1)[:, substrate.subdomain_slice[1]]
                    r = np.sqrt(x ** 2 + y ** 2)
                    p_analytical[r < a] = p0 * np.sqrt(1 - (r[r < a] / a) ** 2)
                else:
                    x = np.array([], dtype=p_analytical.dtype)
                    y = np.array([], dtype=p_analytical.dtype)
                    r = np.array([], dtype=p_analytical.dtype)

                # import matplotlib.pyplot as plt
                # plt.subplot(1,3,1)
                # plt.pcolormesh(p_analytical-p_numerical)
                # plt.colorbar()
                # plt.plot(x, np.sqrt(self.r_s**2-x**2)-(self.r_s-disp0))
                # plt.subplot(1,3,2)
                # plt.pcolormesh(p_analytical)
                # plt.colorbar()
                # plt.subplot(1,3,3)
                # plt.pcolormesh(p_numerical)
                # plt.colorbar()
                # plt.show()
                msg = ""
                msg += "\np_numerical_type:  {}".format(type(p_numerical))
                msg += "\np_numerical_shape: {}".format(p_numerical.shape)
                msg += "\np_numerical_mean:  {}".format(self.pnp.sum(p_numerical) / (nx * ny * 4))
                msg += "\np_numerical_dtype: {}".format(p_numerical.dtype)
                msg += "\np_numerical_max:   {}".format(self.pnp.max(p_numerical))
                msg += "\np_analytical_max:  {}".format(self.pnp.max(p_analytical))
                msg += "\nslice_size:        {}".format((r < .99 * a).sum())
                msg += "\ncontact_radius a:  {}".format(a)
                msg += "\ncomputed normal_force:      {}".format(comp_normal_force)
                msg += "\n{}".format(self.pnp.max(result.jac) - self.pnp.min(result.jac))

                if DEBUG:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 3)

                    plt.colorbar(ax[0].pcolormesh(p_analytical - p_numerical), ax=ax[0])
                    ax[0].set_title("difference")
                    plt.colorbar(ax[1].pcolormesh(p_analytical), ax=ax[1])
                    ax[1].set_title("analytical")
                    plt.colorbar(ax[2].pcolormesh(p_numerical), ax=ax[2])
                    ax[2].set_title("numerical")

                    for ai in ax:
                        ai.set_aspect("equal")
                    # print(MPI.COMM_WORLD.Get_size())
                    fig.savefig("Hertz_plot_proc_%i.png" % self.pnp.comm.Get_rank())

                try:
                    self.assertLess(self.pnp.max(np.abs(p_analytical[r<0.99*a]-
                                        p_numerical[r<0.99*a]))/self.E_s, 1e-3,
                                    msg)




                    # assert the Contact area is OK



                except ValueError as err:
                    msg = str(err) + msg
                    raise ValueError(msg)

suite = unittest.TestSuite([unittest.TestLoader().loadTestsFromTestCase(HertzTest)])

if __name__ in  ['__main__','builtins']:
    print("Running unittest MPI_HertzTest")
    result = unittest.TextTestRunner().run(suite)
