import numpy as np

import pytest

from PyCo.ContactMechanics import Dugdale
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace, FreeFFTElasticHalfSpace
from PyCo.System import make_system
from PyCo.Tools.Logger import screen
from PyCo.Tools.Optimisation import constrained_conjugate_gradients
from PyCo.Topography import Topography, make_sphere


@pytest.mark.skip
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
                                          verbose=True, maxiter=50, logger=screen)
    assert sol.success

    assert (- sol.jac == sigma0).all


def test_sphere():
    nx, ny = (256, 256)
    sx, sy = (2 * np.pi, 2 * np.pi)

    halfspace = FreeFFTElasticHalfSpace((nx, ny), 10, (sx, sx))
    interaction = Dugdale(1, 0.2)
    topography = make_sphere(10, (nx, ny), (sx, sy))
    system = make_system(halfspace, interaction, topography)

    for offset in [0.0, 0.1, 0.2]:
        opt = system.minimize_proxy(
            verbose=True,
            maxiter=50,
            prestol=1e-4,
            offset=offset
        )
        assert opt.success

