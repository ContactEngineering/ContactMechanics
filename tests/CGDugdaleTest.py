import numpy as np

from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import Topography
from PyCo.Tools.Optimisation import constrained_conjugate_gradients


def test_flat():
    nx, ny = (10, 10)
    sx, sy = 1, 1
    Es = 1
    topography = Topography(np.zeros((nx, ny)), (sx, sy))
    hs = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sy))

    sigma0 = 1e-4

    sol = constrained_conjugate_gradients(hs, topography=topography,
                                          Dugdale=(sigma0, 1),
                                          offset=-0.5,
                                          disp0=topography.heights() + 1,
                                          verbose=True, maxiter=50)
    assert sol.success

    assert (- sol.jac == sigma0).all
