#
# Contact area as a function of load for a randomly rough surface
#
# For small loads, the fractional contact area grows linearly with the mean
# pressure, with a slope of the order of 2/rms slope. (Previously
# snippets/plot_area_vs_load.py, which read the output of the command line
# tool.)
#

import matplotlib.pyplot as plt
import numpy as np
from SurfaceTopography.Generation import fourier_synthesis

import ContactMechanics  # noqa: F401 (registers the contact_mechanics pipeline function)

np.random.seed(1)
rms_slope = 0.1
topography = fourier_synthesis((128, 128), (1., 1.), hurst=0.8, rms_slope=rms_slope, short_cutoff=0.05)

# Pressures are in units of the contact modulus E*
mean_pressure, fractional_area, offset, mean_gap, converged = topography.contact_mechanics(nsteps=10)
assert np.all(converged)

order = np.argsort(mean_pressure)
plt.figure()
plt.loglog(mean_pressure[order], fractional_area[order], 'kx-', label='calculation')
p = np.logspace(np.log10(mean_pressure[mean_pressure > 0].min()), np.log10(mean_pressure.max()))
plt.loglog(p, 2 * p / rms_slope, 'r--', label='2 p / (E* rms slope)')
plt.xlabel('mean pressure $p/E^*$')
plt.ylabel('fractional contact area')
plt.legend()
plt.show()
