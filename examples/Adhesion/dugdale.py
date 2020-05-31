#
# Copyright 2019 Lars Pastewka
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

import numpy as np

import matplotlib.pyplot as plt

import PyCo.Adhesion.Adhesion.ReferenceSolutions.MaugisDugdale as MD
from PyCo.ContactMechanics import make_system
from PyCo.Adhesion import Dugdale
from PyCo.ContactMechanics import FreeFFTElasticHalfSpace
from PyCo.ContactMechanics.Tools.Logger import screen
from PyCo.SurfaceTopography import make_sphere

# TODO: this works for exactly these conditions but it sometimes locks for example at (64,64)
nx, ny = (256, 256)
sx, sy = (2 * np.pi, 2 * np.pi)

Es = 10

h0 = 0.2

sigma0 = 1

interaction = Dugdale(sigma0, h0)
if True:
    R = 10
    halfspace = FreeFFTElasticHalfSpace((nx, ny), Es, (sx, sx))
    topography = make_sphere(R, (nx,ny), (sx, sy), offset=0)
else:
    x = np.arange(nx).reshape(-1, 1) / nx * sx
    y = np.arange(ny).reshape(1, -1) / ny * sy

    heights = np.cos(x) * np.cos(y)
    heights -= np.max(heights)

    topography = Topography(heights, (sx, sy))
    halfspace = PeriodicFFTElasticHalfSpace((nx, ny), Es, (sx, sx))

x = np.arange(nx) / nx * sx
y = np.arange(ny) / ny * sy


######################################### plot
class liveplotter:
    def __init__(self, resolution):
        self.fig, (axpc, axg, axp) = plt.subplots(1, 3, figsize=(12, 4))
        axpc.set_title("pressure")
        axpc.axhline(0)
        axpc.axhline(sigma0)
        axpc.set_ylabel("pressure")

        axgc = axpc.twinx()
        axgc.axhline(0)
        axgc.axhline(h0)
        axgc.set_ylabel("gap")

        # axgc.grid()

        axg.set_aspect(1)
        axp.set_aspect(1)
        axg.set_title("gap")
        axp.set_title("p")

        self.axgc = axgc
        self.axpc = axpc
        self.axp = axp
        self.axg = axg

        self.caxp = plt.colorbar(
            axp.pcolormesh(np.zeros(resolution),
                           rasterized=True), ax=self.axp).ax
        self.caxg = plt.colorbar(axg.pcolormesh(np.zeros(resolution), rasterized=True),
                                 ax=self.axg).ax
        self.fig.show()
        self.fig.tight_layout()

    def ax_init(self):
        self.axp.clear()
        self.axg.clear()
        self.axg.set_title("gap")
        self.axp.set_title("p")

    def __call__(self, it, p_r, g_r, d_scalars):
        self.ax_init()

        plt.colorbar(self.axp.pcolormesh(p_r[:nx, :ny] / topography.area_per_pt, rasterized=True),
                     cax=self.caxp)
        self.caxg.clear()
        plt.colorbar(self.axg.pcolormesh(g_r[:nx, :ny], rasterized=True), cax=self.caxg)

        self.axpc.clear()
        self.axgc.clear()
        lgc, = self.axgc.plot(g_r[:nx, ny // 2], "--")
        lpc, = self.axpc.plot(
            (p_r * (p_r <= sigma0))[:nx, ny // 2] / topography.area_per_pt, "+-r", )
        self.axpc.set_ylim(-3, 2)
        self.axgc.set_ylim(-0.5, 2)

        self.fig.canvas.draw()
        plt.pause(0.1)


#########################################

system = make_system(halfspace, interaction, topography)

contact_radiuses = []
forces = []
for offset in [-0.1, 0.0, 0.1, 0.2]:
    sol = system.minimize_proxy(
        #external_force=2,
        verbose=True,
        maxiter=200,
        prestol=1e-4,
        #callback=liveplotter(resolution=topography.nb_grid_pts),
        logger=screen,
        offset=offset
    )

    force = sol.jac
    gap = sol.x[:nx, :ny] - topography.heights() - sol.offset
    contacting_points = sol.active_set #gap <= 1e-4

    contact_radius = np.sqrt(contacting_points.sum() * halfspace.area_per_pt / np.pi)
    contact_radiuses += [contact_radius]
    forces += [force.sum()]

md_contact_radiuses = np.linspace(np.min(contact_radiuses), np.max(contact_radiuses), 101)
md_forces, md_disps = MD.load_and_displacement(md_contact_radiuses, R,
                                               Es, sigma0 * h0, sigma0)

fix, ax = plt.subplots(1,1)
ax.plot(forces, contact_radiuses, 'ro')
ax.plot(md_forces, md_contact_radiuses, 'k-')
#print(contact_radius)
#print(md_force, force.sum())

if True:
    fig, (axp, axg, axc) = plt.subplots(3, 1)
    axp.set_title("pressures, cut")
    axp.plot(x, force[:nx, ny // 2]/halfspace.area_per_pt, "+")
    axp.axhline(-sigma0)
    axp.grid()

    axg.set_title("gap, cut")
    axg.plot(x, gap[:nx, ny // 2], "+")
    axg.axhline(h0)
    axg.grid()

    axc.set_title("contact, cut")
    axc.plot(x, contacting_points[:nx, ny // 2], "+")
    axc.grid()

    plt.show()
