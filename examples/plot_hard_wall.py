#! /usr/bin/env python3

import matplotlib.pyplot as plt
from netCDF4 import Dataset

###

data = Dataset('traj.nc')
plt.figure()
plt.plot(data['load'], data['area'], 'kx')
plt.xlabel('load')
plt.ylabel('area')
plt.show()