from PyCo.Topography.Generation import fourier_synthesis
import pytest
import numpy as np
from NuMPI import MPI

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
    assert abs(topography.rms_slope() - rms_slope) / rms_slope < 1e-1


def test_fourier_synthesis_rms_height_more_wavevectors(comm_self):
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(50):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       rolloff=0,
                                       long_cutoff=s / 8,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n)
                                        )

        realised_rms_heights.append(topography.rms_height())
    # print(abs(np.mean(realised_rms_heights) - rms_height) / rms_height)
    assert abs(np.mean(realised_rms_heights) - rms_height) / rms_height < 0.1  # TODO: this is not very accurate !


def test_fourier_synthesis_rms_height():
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(50):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       long_cutoff=None,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n)
                                       )
        realised_rms_heights.append(topography.rms_height())
    assert abs(np.mean(realised_rms_heights) - rms_height) / rms_height < 0.5  # TODO: this is not very accurate !


def test_fourier_synthesis_c0():
    H = 0.7
    c0 = 8.

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
        ax.loglog(q, c0 * q ** (-2 - 2 * H), "--", label=r"$c_0 q^{-2-2H}$")

        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{iso}$")
        ax.legend()
        ax.set_ylim(bottom=1)
        plt.show(block=True)

        q, psd = topography.power_spectrum_1D()
        fig, ax = plt.subplots()
        ax.loglog(q, psd, label="generated data")
        ax.loglog(q, c0 / np.pi * q ** (-1 - 2 * H), "--", label=r"$c_0 q^{-1-2H}$")

        ax.legend()
        ax.set_xlabel("q")
        ax.set_ylabel(r"$C^{1D}$")
        plt.show(block=True)

@pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="linescans are not supported in MPI programs")
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

@pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="linescans are not supported in MPI programs")
@pytest.mark.parametrize("n", (256, 1024))
def test_fourier_synthesis_linescan_c0(n):
    H = 0.7
    c0 = 8.

    s = n * 4.
    ls = 32
    qs = 2 * np.pi / ls
    np.random.seed(0)
    t = fourier_synthesis(
        (n,), (s,),
        c0=c0,
        hurst=H,
        long_cutoff=s / 2,
        short_cutoff=ls,
        amplitude_distribution=lambda n: np.ones(n)
    )

    if False:
        import matplotlib.pyplot as plt
        q, psd = t.power_spectrum_1D()

        fig, ax = plt.subplots()
        ax.plot(q, psd)
        ax.plot(q, c0 * q ** (-1 - 2 * H))

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylim(bottom=1)
        fig.show()

    ref_slope = np.sqrt(1 / (2 * np.pi) * c0 / (1 - H) * qs ** (2 - 2 * H))
    assert abs(t.rms_slope() - ref_slope) / ref_slope < 1e-1

@pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="linescans are not supported in MPI programs")
def test_fourier_synthesis_linescan_hprms():
    H = 0.7
    hprms = .2

    n = 2048
    s = n * 4.
    ls = 64
    qs = 2 * np.pi / ls
    # np.random.seed(0)
    realised_rms_slopes = []
    for i in range(20):
        t = fourier_synthesis((n,), (s,),
                              rms_slope=hprms,
                              hurst=H,
                              long_cutoff=s / 2,
                              short_cutoff=ls,
                              )
        realised_rms_slopes.append(t.rms_slope())
    ref_slope = hprms
    assert abs(np.mean(realised_rms_slopes) - ref_slope) / ref_slope < 1e-1

@pytest.mark.skipif(MPI.COMM_WORLD.Get_size()> 1,
        reason="linescans are not supported in MPI programs")
def test_fourier_synthesis_linescan_hrms_more_wavevectors():
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    H = 0.7
    hrms = 4.
    n = 4096
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    realised_rms_heights = []
    for i in range(50):
        t = fourier_synthesis((n,), (s,),
                              rms_height=hrms,
                              hurst=H,
                              rolloff=0,
                              long_cutoff=s/8,
                              short_cutoff=ls,
                              )
        realised_rms_heights.append(t.rms_height())
    realised_rms_heights = np.array(realised_rms_heights)
    ref_height = hrms
    #print(np.sqrt(np.mean((realised_rms_heights - np.mean(realised_rms_heights))**2)))
    assert abs(np.mean(realised_rms_heights) - ref_height) / ref_height < 0.1  #
