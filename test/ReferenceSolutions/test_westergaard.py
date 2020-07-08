from ContactMechanics.ReferenceSolutions import Westergaard
import numpy as np


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
