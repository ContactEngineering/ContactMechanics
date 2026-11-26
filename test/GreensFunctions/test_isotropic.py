import numpy as np

from ContactMechanics.GreensFunctions import IsotropicGreensFunction


def test_small_wavevector():
    # Small but finite wavevectors have a finite stiffness (previously zero
    # below an absolute cutoff of 1e-6)
    gf = IsotropicGreensFunction(1.0, 0.3)
    q = 1e-9
    np.testing.assert_allclose(gf.stiffness(q, 0.0), q * gf.stiffness(1.0, 0.0))


def test_zero_wavevector():
    gf = IsotropicGreensFunction(1.0, 0.3)
    np.testing.assert_array_equal(gf.stiffness(0.0, 0.0), 0)
    assert np.all(np.isinf(gf.greens_function(0.0, 0.0)))
