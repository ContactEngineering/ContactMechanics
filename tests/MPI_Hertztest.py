try:
    import unittest
    import numpy as np
    import time
    import math
    from PyCo.ContactMechanics import HardWall
    from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
    from PyCo.SolidMechanics import FreeFFTElasticHalfSpace
    from PyCo.Topography import Sphere
    from PyCo.System import SystemFactory
    #from PyCo.Tools.Logger import screen
    from PyCo.ReferenceSolutions.Hertz import (radius_and_pressure,
                                               surface_displacements,
                                               surface_stress)
    from FFTEngine import PFFTEngine
    from PyCo.Tools.ParallelNumpy import ParallelNumpy

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
        self.pnp = ParallelNumpy()

    def test_elastic_solution(self):
        r = np.linspace(0, self.r_s, 6)/self.r_c
        u = surface_displacements(r) / (self.p_0/self.E_s*self.r_c)
        sig = surface_stress(r)[0]/self.p_0

    def test_constrained_conjugate_gradients(self):
        for kind in ['ref']: # Add 'opt' to test optimized solver, but does
                             # not work on Travis!
            for nx, ny in [(256, 256), (256, 255), (255, 256)]:
                for disp0, normal_force in [(0.1, None), (0, 15.0)]:
                    sx = 5.0

                    substrate = FreeFFTElasticHalfSpace((nx, ny), self.E_s,
                                                        (sx, sx), fftengine=PFFTEngine)

                    interaction = HardWall()
                    surface = Sphere(self.r_s, (nx, ny), (sx, sx))
                    system = SystemFactory(substrate, interaction, surface)

                    result = system.minimize_proxy(offset=disp0,
                                                   external_force=normal_force,
                                                   kind="ref")
                    disp = result.x
                    forces = -result.jac
                    converged = result.success
                    self.assertTrue(converged)


                    if normal_force is not None:
                        self.assertAlmostEqual(normal_force, self.pnp.sum(forces.sum))
                    normal_force = -self.pnp.sum(forces)
                    a, p0 = radius_and_pressure(normal_force, self.r_s,
                                                self.E_s)

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
                    msg += "\nnormal_force:      {}".format(normal_force)
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
                    except ValueError as err:
                        msg = str(err) + msg
                        raise ValueError(msg)

if __name__ == '__main__':
    unittest.main()

