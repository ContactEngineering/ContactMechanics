#! /usr/bin/env python3

"""
Two contacting rough surfaces. Top surface is continuously displaced in
x-direction at constant height.
"""

import numpy as np

from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Topography import read_matrix, TranslatedTopography, CompoundTopography
from PyCo.System import make_system
#from PyCo.Tools import compute_rms_height
from PyCo.Tools.Logger import screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

import matplotlib.pyplot as plt


###

# This is the elastic contact modulus, E*.
E_s = 2

###

# This is the physical size of the surfaces.
sx, sy = 1, 1

# Read the two contacting surfacs.
surface1 = read_matrix('surface1.out', size=(sx, sy))
surface2 = read_matrix('surface2.out', size=(sx, sy))

print('RMS heights of surfaces = {} {}'.format(surface1.rms_height(),
                                               surface2.rms_height()))

# This is the grid resolution of the two surfaces.
nx, ny = surface1.shape

# TranslatedSurface knows how to translate a surface into some direction.
translated_surface1 = TranslatedTopography(surface1)

# This is the compound of the two surfaces, effectively creating a surface
# that is the difference between the two profiles.
compound_surface = CompoundTopography(translated_surface1, surface2)

# Periodic substrate and hard-wall interactions.
substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sx))
interaction = HardWall()

# This creates a "System" object that knows about substrate, interaction and
# surface.
system = make_system(substrate, interaction, compound_surface)

# Initial displacement field.
disp = np.zeros(surface1.shape)

# Dump some information to this NetCDF file. Inspect the NetCDF with the
# 'ncdump' command.
container = NetCDFContainer('traj.nc', mode='w', double=True)
# NetCDF needs to know the resolution/shape
container.set_shape(surface2)
# This creates a field called 'surface2' inside the NetCDF file.
container.surface2 = surface2.array()

# Loop over nd displacement steps.
nd = 3
step_size = 24
for i, c in zip(range(0, step_size*nd, step_size), ['r', 'g', 'b', 'y']):
    print(i)
    xshift = i
    # Set the offset of the translated surface, i.e. the translation vector
    # with respect to the untranslated surface.
    translated_surface1.set_offset(xshift, 0)

    # Solve the contact problem for a constant relative displacement (offset).
    opt = system.minimize_proxy(offset=-0.012, disp0=disp)
    # Alternative: Solve it with external force as boundary conditions
    #opt = system.minimize_proxy(external_force=0.01, disp0=disp)
    disp = opt.x # This is the displacement field
    forces = opt.jac # These are the forces/pressures
    offset = opt.offset # The relative displacement of the two surfaces

    # Note: Either force or offset printed in screen should correspond to what
    # you have specified above.
    print('Total force: {}, Offset: {}'.format(forces.sum(), offset))

    # Dump the information to the NetCDF file.
    frame = container.get_next_frame()
    frame.translated_surface1 = translated_surface1.array()
    frame.displacements = disp
    frame.forces = disp

    # Reconstruct the deformed surfaces for plotting. Note that here we assume
    # that the two moduli are the same and half of the displacement is
    # carried by the top and the other half by the bottom surface.
    top_surface = -translated_surface1[:,:]+disp/2-offset
    bottom_surface = surface2[:,:]-disp/2

    plt.plot(np.arange(nx), top_surface[:,150], c+'-')
    plt.plot(np.arange(nx), bottom_surface[:,150], c+'-')

container.close()

plt.show()