#
# Mean size of contact patches of a randomly rough surface
#
# Contact is computed at a few loads; the contact map is decomposed into
# connected patches (nearest-neighbor connectivity, without periodic
# wrap-around) and the mean number of pixels per patch is reported.
# (Previously snippets/eval_drep.py, which read the forces from the NetCDF
# output of the command line tool.)
#

import numpy as np
from scipy.ndimage import label
from SurfaceTopography.Generation import fourier_synthesis

import ContactMechanics  # noqa: F401 (registers the contact_mechanics pipeline function)

np.random.seed(1)
topography = fourier_synthesis((128, 128), (1., 1.), hurst=0.8, rms_slope=0.1, short_cutoff=0.05)


def report_patch_size(displacement, gap, pressure, contacting_points, offset, mean_pressure, fractional_area):
    patch_numbers, nb_patches = label(contacting_points)
    pixels_per_patch = np.bincount(np.ravel(patch_numbers))[1:]
    print(f'mean pressure = {mean_pressure:.3e} E*, fractional area = {fractional_area:.4f}, '
          f'{nb_patches} patches, mean patch size = {pixels_per_patch.mean():.2f} pixels')


topography.contact_mechanics(pressures=[0.001, 0.003, 0.01], results_callback=report_patch_size)
