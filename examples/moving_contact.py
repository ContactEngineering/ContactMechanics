#! /usr/bin/env python3

"""
Two contacting rough surfaces. Top surface is continuously displaced in
x-direction at constant height.
"""

import numpy as np

from PyCo.ContactMechanics import HardWall
from PyCo.SolidMechanics import PeriodicFFTElasticHalfSpace
from PyCo.Surface import read_matrix, TranslatedSurface, CompoundSurface
from PyCo.System import SystemFactory
#from PyCo.Tools import compute_rms_height
from PyCo.Tools.Logger import screen
from PyCo.Tools.NetCDF import NetCDFContainer

###

import matplotlib.pyplot as plt


###

zshift = -0.012
E_s = 2

###

sx, sy = 1, 1

surface1 = read_matrix('surface1.out', size=(sx, sy))
surface2 = read_matrix('surface2.out', size=(sx, sy))

print('RMS heights of surfaces = {} {}', surface1.compute_rms_height(),
      surface2.compute_rms_height())

nx, ny = surface1.shape

translated_surface1 = TranslatedSurface(surface1)
compound_surface = CompoundSurface(translated_surface1, surface2)

substrate = PeriodicFFTElasticHalfSpace((nx, ny), E_s, (sx, sx))
interaction = HardWall()
system = SystemFactory(substrate, interaction, compound_surface)

disp = np.zeros(surface1.shape)

container = NetCDFContainer('traj.nc', mode='w', double=True)
container.set_shape(surface2)
container.surface2 = surface2.profile()

for i in range(nx):
    print(i)
    xshift = i
    translated_surface1.set_offset(xshift, 0)

    opt = system.minimize_proxy(zshift, disp0=disp)
    disp = opt.x
    forces = opt.jac

    frame = container.get_next_frame()
    frame.translated_surface1 = translated_surface1.profile()
    frame.displacements = disp
    frame.forces = disp

    top_surface = -translated_surface1[:,:]+disp/2-zshift
    bottom_surface = surface2[:,:]-disp/2

    plt.plot(np.arange(nx), top_surface[:,150], 'k-')
    plt.plot(np.arange(nx), bottom_surface[:,150], 'k-')

container.close()