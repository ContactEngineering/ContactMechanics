
import numpy as np

from PyCo.Topography import make_sphere
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace

def test_sphere(comm, fftengine_class):
    nx = 8
    ny = 5
    sx = 6.
    sy = 7.
    R = 20.
    center = (3.,3.)
    substrate = FreeFFTElasticHalfSpace(resolution=(nx, ny), young=1.,
                                        size=(sx, sy),
                                        fftengine=fftengine_class((2*nx, 2*ny),
                                                                  comm=comm))
    extended_topography = make_sphere(R, (2*nx, 2*ny), (sx, sy), centre=center,
                                      subdomain_resolution=substrate.subdomain_resolution,
                                      subdomain_location=substrate.subdomain_location,
                                      pnp=substrate.pnp)
    X, Y, Z = extended_topography.positions_and_heights()

    np.testing.assert_allclose((X-center[0])**2 + (Y-center[1])**2 + (R+Z)**2,  R**2)


def test_sphere_periodic(comm, fftengine_class):
    nx = 8
    ny = 5
    sx = 6.
    sy = 7.
    R = 20.
    center = (1., 1.5)
    substrate = PeriodicFFTElasticHalfSpace(resolution=(nx, ny), young=1.,
                                        size=(sx, sy),
                                        fftengine=fftengine_class(
                                            (nx, ny),
                                            comm=comm))

    extended_topography = make_sphere(R, (nx, ny), (sx, sy),
                                      centre=center,
                                      subdomain_resolution=substrate.subdomain_resolution,
                                      subdomain_location=substrate.subdomain_location,
                                      pnp=substrate.pnp,
                                      periodic=True)

    X, Y, Z = extended_topography.positions_and_heights()

    np.testing.assert_allclose((X - np.where(X < center[0] + sx/2, center[0], center[0] + sx) ) ** 2
                + (Y - np.where(Y < center[1] + sy/2 , center[1], center[1] + sy) ) ** 2
                  + (R + Z) ** 2, R**2)

def test_sphere_standoff(comm, fftengine_class):
    nx = 8
    ny = 5
    sx = 6.
    sy = 7.
    R = 2.
    center = (3., 3.)

    standoff = 10.

    substrate = FreeFFTElasticHalfSpace(resolution=(nx, ny), young=1.,
                                        size=(sx, sy),
                                        fftengine=fftengine_class(
                                            (2 * nx, 2 * ny),
                                            comm=comm))
    extended_topography = make_sphere(R, (2 * nx, 2 * ny), (sx, sy),
                                      centre=center,
                                      subdomain_resolution=substrate.subdomain_resolution,
                                      subdomain_location=substrate.subdomain_location,
                                      pnp=substrate.pnp,
                                      standoff=standoff)
    X, Y, Z = extended_topography.positions_and_heights()

    sl_inner= (X - center[0]) ** 2 + (Y - center[1]) ** 2 < R**2
    np.testing.assert_allclose((
        (X - center[0]) ** 2 +
        (Y - center[1]) ** 2 +
        (R + Z) ** 2)[sl_inner]
        ,  R ** 2)

    np.testing.assert_allclose(Z[np.logical_not(sl_inner)] , - R - standoff )


#def test_paraboloid(comm, fftengine_class)

