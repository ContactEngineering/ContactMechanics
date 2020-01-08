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
