#! /usr/bin/env python3

import matplotlib.pyplot as plt

from PyCo.Surface import read

###

parser = ArgumentParser(description='Plot a 2D surface map (pressure, gap, '
                                    'etc.).')
parser.add_argument('filename', metavar='FILENAME', help='name of map file')

###

surface = read(arguments.filename)

###

try:
    sx, sy = surface.size
except ValueError:
    sx, sy = surface.shape

fig = plt.figure()
ax = fig.add_subplot(111, aspect=sx/sy)
ax.pcolormesh(surface[...])
ax.set_xlim(0, sx)
ax.set_ylim(0, sy)
if surface.unit is not None:
    unit = surface.unit
else:
    unit = 'a.u.'
ax.set_xlabel('Position $x$ ({})'.format(unit))
ax.set_ylabel('Position $y$ ({})'.format(unit))
plt.show()
