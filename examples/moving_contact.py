# %% [markdown]
#
# Copyright 2016, 2018, 2020 Lars Pastewka
#           2019-2020 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# %%
"""
Two contacting rough surfaces. Top surface is continuously displaced in
x-direction at constant height.
"""

# %%
import numpy as np

# %%
from ContactMechanics import PeriodicFFTElasticHalfSpace
from SurfaceTopography import open_topography
from ContactMechanics import make_system
# from PyCo.Tools import compute_rms_height
from ContactMechanics.IO.NetCDF import NetCDFContainer

# %% [markdown]
# ##

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ##

# %%
# This is the elastic contact modulus, E*.
E_s = 2

# %% [markdown]
# ##

# %%
# This is the physical physical_sizes of the surfaces.
sx, sy = 1, 1

# %%
# Read the two contacting surfacs.
surface1 = open_topography('surface1.out',).topography(physical_sizes=(sx, sy), periodic=True)
surface2 = open_topography('surface2.out').topography(physical_sizes=(sx, sy), periodic=True)

# %%
print('RMS heights of surfaces = {} {}'.format(surface1.rms_height_from_area(),
                                               surface2.rms_height_from_area()))

# %%
# This is the grid nb_grid_pts of the two surfaces.
nx, ny = surface1.nb_grid_pts

# %%
# TranslatedSurface knows how to translate a surface into some direction.
translated_surface1 = surface1.translate()

# %%
# This is the compound of the two surfaces, effectively creating a surface
# that is the difference between the two profiles.
compound_surface = translated_surface1.superpose(surface2)

# %%
# Periodic substrate and hard-wall interactions.
substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sx))

# %%
# This creates a "System" object that knows about substrate, interaction and
# surface.
system = make_system(substrate=substrate, surface=compound_surface)

# %%
# Initial displacement field.
disp = np.zeros(surface1.nb_grid_pts)

# %%
# Dump some information to this NetCDF file. Inspect the NetCDF with the
# 'ncdump' command.
container = NetCDFContainer('traj.nc', mode='w', double=True)
# NetCDF needs to know the nb_grid_pts/shape
container.set_shape(surface2.nb_grid_pts)
# This creates a field called 'surface2' inside the NetCDF file.
container.surface2 = surface2.heights()

# %%
# Loop over nd displacement steps.
nd = 3
step_size = 24
for i, c in zip(range(0, step_size * nd, step_size), ['r', 'g', 'b', 'y']):
    print(i)
    xshift = i
    # Set the offset of the translated surface, i.e. the translation vector
    # with respect to the untranslated surface.
    translated_surface1.offset = xshift, 0

    # Solve the contact problem for a constant relative displacement (offset).
    opt = system.minimize_proxy(offset=-0.012, initial_displacements=disp)
    # Alternative: Solve it with external force as boundary conditions
    # opt = system.minimize_proxy(external_force=0.01, disp0=disp)
    disp = opt.x  # This is the displacement field
    forces = opt.jac  # These are the forces/pressures
    offset = opt.offset  # The relative displacement of the two surfaces

    # Note: Either force or offset printed in screen should correspond to what
    # you have specified above.
    print('Total force: {}, Offset: {}'.format(forces.sum(), offset))

    # Dump the information to the NetCDF file.
    frame = container.get_next_frame()
    frame.translated_surface1 = translated_surface1.heights()
    frame.displacements = disp
    frame.forces = disp

    # Reconstruct the deformed surfaces for plotting. Note that here we assume
    # that the two moduli are the same and half of the displacement is
    # carried by the top and the other half by the bottom surface.
    top_surface = -translated_surface1[:, :] + disp / 2 - offset
    bottom_surface = surface2[:, :] - disp / 2

    plt.plot(np.arange(nx), top_surface[:, 150], c + '-')
    plt.plot(np.arange(nx), bottom_surface[:, 150], c + '-')

# %%
container.close()

# %%
plt.show()

# %%
