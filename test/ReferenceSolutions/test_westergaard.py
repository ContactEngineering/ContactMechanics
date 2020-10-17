from ContactMechanics.ReferenceSolutions import Westergaard
import numpy as np

from NuMPI import MPI
import pytest

pytestmark = pytest.mark.skipif(MPI.COMM_WORLD.Get_size() > 1,
                                reason="tests only serial funcionalities, "
                                       "please execute with pytest")


def test_mean_displacements():
    a = 0.3

    analytical = Westergaard.mean_displacement(a)

    if False:
        import matplotlib.pyplot as plt
        ns = [10, 100, 1000, 10000]
        plt.loglog(
            ns,
            [(np.mean(Westergaard.displacements(np.linspace(0, 1, n), a))
              - analytical) / abs(analytical) for n in ns])
        plt.show(block=True)
    assert abs(
        np.mean(Westergaard.displacements(np.linspace(0, 1, 1000), a))
        - analytical) / abs(analytical) < 1e-3


def test_radius_mean_pressure_inverse():
    radius = 0.4

    assert abs(radius - Westergaard.contact_radius(
        Westergaard.mean_pressure(radius))) < 1e-13
