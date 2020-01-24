from PyCo.Topography.Generation import fourier_synthesis
import pytest
import numpy as np


@pytest.mark.parametrize("n", [128, 129])
def test_fourier_synthesis(n):
    H = 0.74
    rms_slope = 1.2
    qs = 2 * np.pi / (0.4e-9)  # 1/m
    s = 2e-6

    topography = fourier_synthesis((n, n), (s, s),
                                   H,
                                   rms_slope=rms_slope,
                                   long_cutoff=s / 4,
                                   short_cutoff=4 * s / n)

    qx, psdx = topography.power_spectrum_1D()
    qy, psdy = topography.transpose().power_spectrum_1D()

    assert psdy[-1] < 10 * psdx[-1]  # assert psdy is not much bigger


def test_fourier_synthesis_c0():
    H = 0.7
    c0 = 1.

    n = 512
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    topography = fourier_synthesis((n, n), (s, s),
                                   H,
                                   c0=c0,
                                   long_cutoff=s / 2,
                                   short_cutoff=ls,
                                   amplitude_distribution=lambda n: np.ones(n)
                                   )
    ref_slope = np.sqrt(1 / (4 * np.pi) * c0 / (1 - H) * qs ** (2 - 2 * H))
    assert abs(topography.rms_slope() - ref_slope) / ref_slope < 1e-1

    if False:
        import matplotlib.pyplot as plt
        q, psd = topography.power_spectrum_2D()

        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="generated data")
        ax.loglog(q, c0 * q ** (-2 - 2 * H), label=r"$c_0 q^{-2-2H}$")

        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{iso}$")
        ax.legend()
        plt.show(block=True)

        q, psd = topography.power_spectrum_1D()
        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="generated data")
        ax.loglog(q, c0 / np.pi * q ** (-1 - 2 * H), label=r"$c_0 q^{-1-2H}$")

        ax.legend()
        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{1D}$")
        plt.show(block=True)

def test_fourier_synthesis_1D_input():
    H = 0.7
    c0 = 1.

    n = 512
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    topography = fourier_synthesis((n,), (s,),
                                   H,
                                   c0=c0,
                                   long_cutoff=s / 2,
                                   short_cutoff=ls,
                                   amplitude_distribution=lambda n: np.ones(n)
                                   )

