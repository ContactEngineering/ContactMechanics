
import numpy as np

from PyCo.Topography import make_sphere
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace

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
    extended_topography = make_sphere(R, (2*nx, 2*ny), (2*sx, 2*sy), centre=center,
                                      subdomain_resolution=substrate.subdomain_resolution,
                                      subdomain_location=substrate.subdomain_location,
                                      pnp=substrate.pnp)
    X, Y, Z = extended_topography.positions_and_heights()
    np.testing.assert_allclose((X-center[0])**2 + (Y-center[1])**2 + (R+Z)**2,  R**2)

#def test_paraboloid(comm, fftengine_class)

