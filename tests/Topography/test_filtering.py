from PyCo.Topography.Generation import fourier_synthesis
import numpy as np

def test_lowcut():
    n = 200 # high number of points required because of binning in the isotropic psd
    #t = Topography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13,13), 0.9, 1.)

    q_l = 2 * np.pi / 13 * n / 4
    q, psd = t.lowcut(q_l=q_l).power_spectrum_2D()
    assert (psd[q < 0.9 * q_l] < 1e-10).all()
    # the cut is not clean because of the binning in the 2D PSD (Ciso)

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.loglog(q, psd )
        ax.loglog(*t.power_spectrum_2D())
        ax.axvline(q_l)
        fig.show()


def test_highcut():
    n = 100
    #t = Topography(np.zeros(n,n), (2,3))
    t = fourier_synthesis((n, n), (13,13), 0.9, 1.)

    q_s= 2 * np.pi / 13  * 0.4 * n
    q, psd = t.highcut(q_s=q_s).power_spectrum_2D()
    assert (psd[q > 1.5 *  q_s] < 1e-10).all()

    if False:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.loglog(q, psd , label="filtered")
        ax.loglog(*t.power_spectrum_2D(), label="original")
        ax.legend()
        fig.show()