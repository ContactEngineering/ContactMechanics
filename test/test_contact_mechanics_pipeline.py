"""
Tests of the `contact_mechanics` pipeline function.
"""

import numpy as np
import pytest
from SurfaceTopography import Topography

import ContactMechanics  # noqa: F401


def _rough_topography(shift=0., seed=1, nb_grid_pts=64):
    rng = np.random.default_rng(seed)
    q = np.fft.fftfreq(nb_grid_pts) * nb_grid_pts
    abs_q = np.sqrt(q.reshape(-1, 1) ** 2 + q.reshape(1, -1) ** 2)
    abs_q[0, 0] = 1
    amplitude = abs_q ** -1.8
    amplitude[0, 0] = 0
    amplitude[abs_q > nb_grid_pts / 4] = 0
    heights = np.fft.ifft2(amplitude * np.exp(2j * np.pi * rng.random(amplitude.shape))).real
    return Topography(0.01 * heights / heights.std() + shift, (1., 1.), periodic=True)


@pytest.mark.parametrize('shift', [5., -3.])
def test_automatic_steps_invariant_under_height_shift(shift):
    # Offsets are measured relative to the heights, hence shifting all heights
    # must shift the offsets and leave everything else unchanged
    reference = _rough_topography().contact_mechanics(nsteps=8)
    shifted = _rough_topography(shift).contact_mechanics(nsteps=8)
    for i in [0, 1, 3, 4]:
        np.testing.assert_allclose(shifted[i], reference[i], atol=1e-6)
    np.testing.assert_allclose(shifted[2], reference[2] - shift)
    assert np.all(shifted[4])


def test_automatic_steps_span_contact_areas():
    mean_pressure, fractional_contact_area, mean_displacement, mean_gap, converged = \
        _rough_topography().contact_mechanics(nsteps=8)
    assert np.all(converged)
    assert fractional_contact_area.max() <= 1
    # The last step extrapolated towards full contact
    assert fractional_contact_area.max() > 0.5
    assert fractional_contact_area.min() < 0.01


def test_pressures():
    pressures = [0.001, 0.01]
    mean_pressure, fractional_contact_area, mean_displacement, mean_gap, converged = \
        _rough_topography().contact_mechanics(pressures=pressures)
    assert np.all(converged)
    np.testing.assert_allclose(mean_pressure, pressures)
    assert np.all(np.diff(fractional_contact_area) > 0)


def test_optimizer_kwargs_do_not_clash_with_explicit_arguments():
    topography = _rough_topography()
    results = topography.contact_mechanics(offsets=[0.], maxiter=1000,
                                           optimizer_kwargs=dict(pentol=1e-8, maxiter=10))
    # Explicit argument wins over optimizer_kwargs
    assert results[4][0]


def test_results_callback_receives_fractional_area():
    received = []

    def callback(displacement, gap, pressure, contacting_points, offset, mean_pressure, fractional_contact_area):
        received.append((contacting_points.mean(), fractional_contact_area, pressure.mean(), mean_pressure))

    _, fractional_contact_area, _, _, _ = _rough_topography().contact_mechanics(nsteps=3, results_callback=callback)
    assert len(received) == 3
    for area_from_map, area, mean_pressure_from_map, mean_pressure in received:
        np.testing.assert_allclose(area_from_map, area)
        np.testing.assert_allclose(mean_pressure_from_map, mean_pressure)
    np.testing.assert_allclose([r[1] for r in received], fractional_contact_area)
