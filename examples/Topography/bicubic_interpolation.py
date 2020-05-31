#
# Copyright 2020 Antoine Sanner
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from PyCo.SurfaceTopography.Interpolation import Bicubic
from PyCo.SurfaceTopography.Generation import fourier_synthesis
import matplotlib.pyplot as plt
import numpy as np
#from muFFT import FourierInterpolation # future


nx, ny = [512] * 2
sx, sy = [1.] * 2



# %% Generate random topography
hc = 0.1 * sx

topography = fourier_synthesis((nx, ny), (sx, sy), 0.8, rms_height=1.,
                                short_cutoff=hc, long_cutoff=hc+1e-9, )
topography = topography.scale(1/topography.rms_height())
dx, dy = topography.fourier_derivative()

# %%
fig, ax = plt.subplots()
ax.imshow(topography.heights())
fig.show()
# %%
fig, ax = plt.subplots()
ax.imshow(dx)
fig.show()
fig, ax = plt.subplots()
ax.imshow(topography.derivative(1)[0])
fig.show()


# %% check bicubic interpolation against fourier interpolation

# %%
fig, ax = plt.subplots()
x, y = topography.positions()
ax.plot(x[:, 0], topography.heights()[:,0], ".k", label="original")


skips = [4, 8, 16, 32, 64]
rms_err = []
max_err = []
for skip in skips:
    grid_slice = (slice(None, None, skip), slice(None, None, skip))

    interp = Bicubic(topography.heights()[grid_slice],
                   dx[grid_slice] * topography.pixel_size[0] * skip,
                   dy[grid_slice] *topography.pixel_size[1] * skip
                   )

    interp_field, interp_derx, interp_dery = interp(x / (topography.pixel_size[0] * skip),
                                                    y / (topography.pixel_size[1] * skip), derivative=1)
    l, = ax.plot(x[grid_slice][:,0], topography.heights()[grid_slice][:, 0], "+")
    ax.plot(x[:, 0], interp_field[:, 0], color=l.get_color(), label=r"bicubic, $l_{cor} / \Delta_x=$"+ f"{hc / (skip * topography.pixel_size[0])}")

    rms_err.append( np.sqrt(np.mean((interp_field - topography.heights())**2 )) )
    max_err.append( np.max(abs(interp_field - topography.heights())) )
    ax.legend()
    fig.show()

skips = np.array(skips)
rms_err = np.array(rms_err)
max_err = np.array(max_err)

# %%

fig, ax = plt.subplots()
sampling = (skips * topography.pixel_size[0]) / hc

ax.plot(sampling, rms_err, "-o", label="rms error")
ax.plot(sampling, max_err, "-o", label="max error")

ax.set_xlabel(r"$\Delta_x / l_{cor}$")
ax.legend()
ax.set_yscale("log")
ax.set_xscale("log")
fig.show()