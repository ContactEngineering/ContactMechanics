import numpy as np
from SurfaceTopography import Topography

from ContactMechanics import make_system


def test_heuristic_pentol():
    # just checks works without bug

    nx, ny = (10,5)

    topo = Topography(np.resize(np.cos(2 * np.pi * np.arange(nx) / nx),
                (nx, ny)), physical_sizes=(nx, ny),  periodic=True)

    system = make_system(surface=topo, substrate="periodic", young=1)
    sol = system.minimize_proxy(offset=-0.5)
