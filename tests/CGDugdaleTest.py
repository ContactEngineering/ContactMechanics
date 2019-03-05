import numpy as np

import matplotlib.pyplot as plt
from PyCo.Topography import make_sphere
from PyCo.SolidMechanics import FreeFFTElasticHalfSpace, PeriodicFFTElasticHalfSpace
from PyCo.Topography import Topography
from PyCo.Tools.Optimisation.ConstrainedConjugateGradientsDugdale import constrained_conjugate_gradients_Dugdale


def test_flat():
    nx, ny = (10,10)
    sx, sy = 1,1
    Es = 1
    topography = Topography(np.zeros((nx, ny)), (sx, sy))
    hs = PeriodicFFTElasticHalfSpace((nx, ny), Es,(sx, sy))

    sigma0 = 1e-4

    sol = constrained_conjugate_gradients_Dugdale(hs, topography=topography,
                                                  sigma0=sigma0,
                                                  h0=1, offset=-0.5,
                                                  disp0= topography.heights() +1 ,
                                                  verbose=True, maxiter=50)
    assert sol.success

    assert (- sol.jac == sigma0).all